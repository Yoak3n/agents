use crate::schema::extension::Skill;

/// Skill 选择器 trait — 子 agent，决定暴露哪些 skills 给主 agent
pub trait SkillSelector: Send + Sync {
    /// 根据用户消息和可用 skills，返回本次应激活的 skills
    fn select(&self, message: &str, skills: &[Skill]) -> Vec<Skill>;
}

/// 基于 read_when 场景匹配的选择器（零开销，默认实现）
pub struct HeuristicSelector;

impl SkillSelector for HeuristicSelector {
    fn select(&self, message: &str, skills: &[Skill]) -> Vec<Skill> {
        if skills.is_empty() {
            return Vec::new();
        }

        let message_lower = message.to_lowercase();

        let mut scored: Vec<(usize, &Skill)> = skills
            .iter()
            .map(|skill| {
                let mut score = 0usize;

                // 匹配 skill name
                if message_lower.contains(&skill.name.to_lowercase()) {
                    score += 10;
                }

                // 匹配 description 中的关键词
                for word in skill.description.to_lowercase().split_whitespace() {
                    if word.len() > 3 && message_lower.contains(word) {
                        score += 2;
                    }
                }

                // 匹配 read_when 场景
                for scenario in &skill.read_when {
                    let scenario_lower = scenario.to_lowercase();
                    // 场景中的关键词匹配
                    for word in scenario_lower.split_whitespace() {
                        if word.len() > 3 && message_lower.contains(word) {
                            score += 3;
                        }
                    }
                }

                (score, skill)
            })
            .collect();

        // 按分数降序排列
        scored.sort_by(|a, b| b.0.cmp(&a.0));

        // 返回分数 > 0 的 skills，无匹配返回空
        scored
            .into_iter()
            .filter(|(score, _)| *score > 0)
            .map(|(_, skill)| skill.clone())
            .collect()
    }
}

impl Default for HeuristicSelector {
    fn default() -> Self {
        Self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_skill(name: &str, desc: &str, read_when: &[&str]) -> Skill {
        Skill {
            name: name.into(),
            description: desc.into(),
            instructions: format!("You are a {name}"),
            read_when: read_when.iter().map(|s| s.to_string()).collect(),
            metadata: None,
            allowed_tools: None,
        }
    }

    #[test]
    fn test_exact_name_match() {
        let skills = vec![
            make_skill("coder", "writes functions", &["programming"]),
            make_skill("reviewer", "reviews pull requests", &["auditing"]),
        ];
        let selector = HeuristicSelector;
        let result = selector.select("help me with the coder please", &skills);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "coder");
    }

    #[test]
    fn test_read_when_match() {
        let skills = vec![
            make_skill("coder", "writes code", &["writing functions"]),
            make_skill("reviewer", "reviews code", &["reviewing code for bugs"]),
        ];
        let selector = HeuristicSelector;
        let result = selector.select("I need a code review", &skills);
        assert!(result.iter().any(|s| s.name == "reviewer"));
    }

    #[test]
    fn test_no_match_returns_empty() {
        let skills = vec![
            make_skill("coder", "writes code", &["programming"]),
            make_skill("reviewer", "reviews code", &["auditing"]),
        ];
        let selector = HeuristicSelector;
        let result = selector.select("hello world", &skills);
        assert!(result.is_empty());
    }
}
