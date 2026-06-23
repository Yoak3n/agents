pub mod embedding;
pub mod openai;
pub mod semaphore;

#[cfg(feature = "storage")]
pub use embedding::embed_missing_documents;
pub use embedding::{
    EmbeddingAdapter, EmbeddingError, OllamaEmbeddingAdapter, OpenAIEmbeddingAdapter,
    create_embedding_adapter,
};
pub use openai::OpenAIAdapter;
pub use semaphore::{RateLimitedAdapter, Semaphore, SemaphoreError};

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::schema::common::ModelProvider;

/// 负载均衡策略 — 内部自动决策，不暴露给用户
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BalanceStrategy {
    RoundRobin,
    WeightedRoundRobin,
}

impl BalanceStrategy {
    /// 根据 provider 配置自动选择策略：
    /// - 单 provider → RoundRobin（无意义但兜底）
    /// - 所有 weight 相同 → RoundRobin
    /// - weight 不同 → WeightedRoundRobin
    pub(crate) fn auto_detect(providers: &[ModelProvider]) -> Self {
        if providers.len() <= 1 {
            return Self::RoundRobin;
        }
        let first_weight = providers[0].weight;
        if providers.iter().all(|p| p.weight == first_weight) {
            Self::RoundRobin
        } else {
            Self::WeightedRoundRobin
        }
    }
}

pub struct ProviderBalancer {
    providers: Vec<ModelProvider>,
    strategy: BalanceStrategy,
    rr_index: AtomicUsize,
    wrr_current_index: AtomicUsize,
}

impl Clone for ProviderBalancer {
    fn clone(&self) -> Self {
        Self {
            providers: self.providers.clone(),
            strategy: self.strategy,
            rr_index: AtomicUsize::new(self.rr_index.load(Ordering::Relaxed)),
            wrr_current_index: AtomicUsize::new(self.wrr_current_index.load(Ordering::Relaxed)),
        }
    }
}

impl ProviderBalancer {
    pub fn new(providers: Vec<ModelProvider>) -> Self {
        let strategy = BalanceStrategy::auto_detect(&providers);
        Self {
            providers,
            strategy,
            rr_index: AtomicUsize::new(0),
            wrr_current_index: AtomicUsize::new(0),
        }
    }

    pub fn select(&self) -> Option<&ModelProvider> {
        let enabled: Vec<&ModelProvider> = self.providers.iter().filter(|p| p.enabled).collect();

        if enabled.is_empty() {
            return None;
        }

        match self.strategy {
            BalanceStrategy::RoundRobin => self.select_round_robin(&enabled),
            BalanceStrategy::WeightedRoundRobin => self.select_weighted_round_robin(&enabled),
        }
    }

    fn select_round_robin<'a>(&self, providers: &[&'a ModelProvider]) -> Option<&'a ModelProvider> {
        let idx = self.rr_index.fetch_add(1, Ordering::Relaxed) % providers.len();
        providers.get(idx).copied()
    }

    fn select_weighted_round_robin<'a>(
        &self,
        providers: &[&'a ModelProvider],
    ) -> Option<&'a ModelProvider> {
        if providers.is_empty() {
            return None;
        }
        if providers.len() == 1 {
            return providers.first().copied();
        }

        let total_weight: u32 = providers.iter().map(|p| p.weight).sum();
        if total_weight == 0 {
            return self.select_round_robin(providers);
        }

        let current_idx = self.wrr_current_index.load(Ordering::Relaxed);

        let mut best_idx = current_idx;
        let mut best_weight = 0u32;

        for i in 0..providers.len() {
            let idx = (current_idx + i) % providers.len();
            let p = providers[idx];
            if p.weight > best_weight {
                best_weight = p.weight;
                best_idx = idx;
            }
        }

        let next_idx = (best_idx + 1) % providers.len();
        self.wrr_current_index.store(next_idx, Ordering::Relaxed);

        providers.get(best_idx).copied()
    }

    pub fn providers(&self) -> &[ModelProvider] {
        &self.providers
    }

    pub fn update_providers(&mut self, providers: Vec<ModelProvider>) {
        self.strategy = BalanceStrategy::auto_detect(&providers);
        self.providers = providers;
        self.rr_index.store(0, Ordering::Relaxed);
        self.wrr_current_index.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::schema::common::ModelKind;

    fn make_provider(name: &str, weight: u32) -> ModelProvider {
        let mut p = ModelProvider::new(ModelKind::Chat, name, "http://localhost", "key", "model");
        p.weight = weight;
        p
    }

    #[test]
    fn test_auto_detect_round_robin() {
        let providers = vec![make_provider("a", 1), make_provider("b", 1)];
        assert_eq!(
            BalanceStrategy::auto_detect(&providers),
            BalanceStrategy::RoundRobin
        );
    }

    #[test]
    fn test_auto_detect_weighted() {
        let providers = vec![make_provider("a", 5), make_provider("b", 1)];
        assert_eq!(
            BalanceStrategy::auto_detect(&providers),
            BalanceStrategy::WeightedRoundRobin
        );
    }

    #[test]
    fn test_round_robin() {
        let providers = vec![
            make_provider("a", 1),
            make_provider("b", 1),
            make_provider("c", 1),
        ];
        let balancer = ProviderBalancer::new(providers);

        let names: Vec<String> = (0..6)
            .filter_map(|_| balancer.select().map(|p| p.name.clone()))
            .collect();
        assert_eq!(names, vec!["a", "b", "c", "a", "b", "c"]);
    }

    #[test]
    fn test_weighted_round_robin() {
        let providers = vec![
            make_provider("a", 5),
            make_provider("b", 1),
            make_provider("c", 1),
        ];
        let balancer = ProviderBalancer::new(providers);

        let mut counts = std::collections::HashMap::new();
        for _ in 0..70 {
            if let Some(p) = balancer.select() {
                *counts.entry(p.name.clone()).or_insert(0) += 1;
            }
        }
        assert!(counts.get("a").unwrap_or(&0) > counts.get("b").unwrap_or(&0));
    }

    #[test]
    fn test_disabled_provider_skipped() {
        let mut p1 = make_provider("a", 1);
        p1.enabled = false;
        let p2 = make_provider("b", 1);
        let balancer = ProviderBalancer::new(vec![p1, p2]);

        for _ in 0..5 {
            assert_eq!(balancer.select().unwrap().name, "b");
        }
    }
}
