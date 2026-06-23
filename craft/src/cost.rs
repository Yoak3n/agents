use std::collections::HashMap;

use async_trait::async_trait;

use yoakore::prelude::*;

/// Per-provider pricing. All prices are per 1000 tokens.
#[derive(Debug, Clone)]
pub struct PricingRule {
    pub input_price_per_1k: f64,
    pub cached_input_price_per_1k: f64,
    pub output_price_per_1k: f64,
}

impl PricingRule {
    pub const fn new(input: f64, output: f64) -> Self {
        Self {
            input_price_per_1k: input,
            cached_input_price_per_1k: input, // default: same as input
            output_price_per_1k: output,
        }
    }

    /// Set a separate cached input price (e.g., Claude prompt caching is 90% cheaper).
    pub const fn cached_input(mut self, price: f64) -> Self {
        self.cached_input_price_per_1k = price;
        self
    }
}

/// Maps provider names to pricing rules, with a fallback default.
///
/// ```rust
/// use yoakraft::cost::{PricingTable, PricingRule};
///
/// let table = PricingTable::new()
///     .default(PricingRule::new(0.001, 0.002))
///     .provider("claude", PricingRule::new(0.003, 0.015).cached_input(0.0003))
///     .provider("deepseek", PricingRule::new(0.00014, 0.00028));
/// ```
#[derive(Debug, Clone)]
pub struct PricingTable {
    rules: HashMap<String, PricingRule>,
    default: PricingRule,
}

impl PricingTable {
    /// Create an empty pricing table with zero-cost defaults.
    pub fn new() -> Self {
        Self {
            rules: HashMap::new(),
            default: PricingRule::new(0.0, 0.0),
        }
    }

    /// Set the default pricing rule (used when no provider-specific rule matches).
    pub fn default(mut self, rule: PricingRule) -> Self {
        self.default = rule;
        self
    }

    /// Add a pricing rule for a specific provider name.
    ///
    /// Matches against `ModelProvider.name` (case-insensitive).
    pub fn provider(mut self, name: impl Into<String>, rule: PricingRule) -> Self {
        self.rules.insert(name.into().to_lowercase(), rule);
        self
    }

    /// Look up the pricing rule for a provider name.
    fn get(&self, provider_name: &str) -> &PricingRule {
        self.rules
            .get(&provider_name.to_lowercase())
            .unwrap_or(&self.default)
    }
}

impl Default for PricingTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Cost calculation trait — defines usage recording and cost query behavior.
///
/// Implement this trait to replace the default cost tracking logic.
#[async_trait]
pub trait CostCalculator: Send + Sync {
    /// Record usage from a single LLM call.
    fn record_usage<'a>(
        &'a self,
        provider: &'a str,
        input: u64,
        cached_input: u64,
        output: u64,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + 'a>>;

    /// Estimated total cost across all providers.
    fn total_cost(&self) -> std::pin::Pin<Box<dyn std::future::Future<Output = f64> + Send + '_>>;

    /// Cost breakdown by provider.
    fn cost_by_provider(
        &self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = HashMap<String, f64>> + Send + '_>>;
}

/// Per-provider accumulated usage.
#[derive(Debug, Clone, Default)]
pub struct ProviderUsage {
    pub input_tokens: u64,
    pub cached_input_tokens: u64,
    pub output_tokens: u64,
    pub requests: u64,
}

impl ProviderUsage {
    /// Sum of all token types (input + cached input + output).
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens + self.cached_input_tokens + self.output_tokens
    }
}

/// Hook that tracks token usage and estimates cost per provider.
///
/// Token counts are estimated from content length (not from API usage data).
/// For accurate tracking, ensure the adapter parses the `usage` field from API responses.
pub struct CostTracker {
    pricing: PricingTable,
    /// Accumulated usage keyed by provider name.
    usage: tokio::sync::RwLock<HashMap<String, ProviderUsage>>,
}

impl CostTracker {
    /// Create a new tracker with the given pricing table.
    pub fn new(pricing: PricingTable) -> Self {
        Self {
            pricing,
            usage: tokio::sync::RwLock::new(HashMap::new()),
        }
    }

    /// Get a snapshot of usage for all providers.
    pub async fn usage(&self) -> HashMap<String, ProviderUsage> {
        self.usage.read().await.clone()
    }

    /// Get usage for a specific provider.
    pub async fn provider_usage(&self, name: &str) -> ProviderUsage {
        self.usage
            .read()
            .await
            .get(name)
            .cloned()
            .unwrap_or_default()
    }

    /// Total tokens across all providers.
    pub async fn total_tokens(&self) -> u64 {
        self.usage
            .read()
            .await
            .values()
            .map(|u| u.total_tokens())
            .sum()
    }

    /// Estimated total cost across all providers.
    pub async fn estimated_cost(&self) -> f64 {
        let usage = self.usage.read().await;
        usage
            .iter()
            .map(|(name, u)| {
                let rule = self.pricing.get(name);
                let input_cost = (u.input_tokens as f64 / 1000.0) * rule.input_price_per_1k;
                let cached_cost =
                    (u.cached_input_tokens as f64 / 1000.0) * rule.cached_input_price_per_1k;
                let output_cost = (u.output_tokens as f64 / 1000.0) * rule.output_price_per_1k;
                input_cost + cached_cost + output_cost
            })
            .sum()
    }

    /// Estimated cost breakdown by provider.
    pub async fn cost_by_provider(&self) -> HashMap<String, f64> {
        let usage = self.usage.read().await;
        usage
            .iter()
            .map(|(name, u)| {
                let rule = self.pricing.get(name);
                let input_cost = (u.input_tokens as f64 / 1000.0) * rule.input_price_per_1k;
                let cached_cost =
                    (u.cached_input_tokens as f64 / 1000.0) * rule.cached_input_price_per_1k;
                let output_cost = (u.output_tokens as f64 / 1000.0) * rule.output_price_per_1k;
                (name.clone(), input_cost + cached_cost + output_cost)
            })
            .collect()
    }
}

#[async_trait]
impl CostCalculator for CostTracker {
    fn record_usage<'a>(
        &'a self,
        provider: &'a str,
        input: u64,
        cached_input: u64,
        output: u64,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + 'a>> {
        Box::pin(async move {
            let mut usage = self.usage.write().await;
            let entry = usage.entry(provider.to_string()).or_default();
            entry.input_tokens += input;
            entry.cached_input_tokens += cached_input;
            entry.output_tokens += output;
            entry.requests += 1;
        })
    }

    fn total_cost(&self) -> std::pin::Pin<Box<dyn std::future::Future<Output = f64> + Send + '_>> {
        Box::pin(async move { self.estimated_cost().await })
    }

    fn cost_by_provider(
        &self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = HashMap<String, f64>> + Send + '_>>
    {
        Box::pin(async move { Self::cost_by_provider(self).await })
    }
}

#[async_trait]
impl AgentHook for CostTracker {
    async fn after_llm_call(
        &self,
        ctx: &HookContext<'_>,
        response: &mut AgentResponse,
    ) -> HookResult {
        let provider_name = &ctx.provider.name;
        let mut usage = self.usage.write().await;
        let entry = usage.entry(provider_name.clone()).or_default();

        if let Some(ref api_usage) = response.usage {
            // Use real usage data from the API response
            entry.input_tokens += api_usage.prompt_tokens;
            entry.cached_input_tokens += api_usage.cached_input_tokens;
            entry.output_tokens += api_usage.completion_tokens;
            log::debug!(
                "CostTracker [{}]: real usage: {} in ({} cached) / {} out",
                provider_name,
                api_usage.prompt_tokens,
                api_usage.cached_input_tokens,
                api_usage.completion_tokens,
            );
        } else {
            // Fallback: estimate output tokens from content
            let output_tokens = match &response.kind {
                AgentResponseKind::MessageComplete(msg) => {
                    yoakore::estimate_tokens(&msg.content) as u64
                }
                AgentResponseKind::ToolCalls(_) => 0,
            };
            entry.output_tokens += output_tokens;
            log::debug!(
                "CostTracker [{}]: estimated +{} out",
                provider_name,
                output_tokens,
            );
        }

        entry.requests += 1;
        HookResult::Continue
    }
}
