use std::path::Path;

use serde::Deserialize;

/// SKILL.md 的 YAML frontmatter（OpenClaw 标准格式）
#[derive(Debug, Deserialize)]
struct SkillFrontmatter {
    /// 显示名称
    name: String,
    /// 一句话描述
    #[serde(default)]
    description: String,
    /// 触发场景列表（选择器根据此判断是否激活）
    #[serde(default, rename = "read_when")]
    read_when: Vec<String>,
    /// 附加元数据（JSON 字符串）
    #[serde(default)]
    metadata: Option<String>,
    /// 允许使用的工具（支持模式匹配，如 "Bash(agent-browser:*)"）
    #[serde(default, rename = "allowed-tools")]
    allowed_tools: Option<String>,
}

/// 运行时 Skill — 从 workspace/skills/ 目录加载（OpenClaw 格式）
#[derive(Debug, Clone)]
pub struct Skill {
    pub name: String,
    pub description: String,
    /// frontmatter 之后的 Markdown body = 完整操作指南
    pub instructions: String,
    pub read_when: Vec<String>,
    pub metadata: Option<String>,
    pub allowed_tools: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum SkillError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("YAML parse error: {0}")]
    Yaml(String),

    #[error("missing SKILL.md in {0}")]
    MissingSkillFile(String),
}

impl Skill {
    /// 从目录加载 skill（读取 SKILL.md）
    ///
    /// SKILL.md 格式：
    /// ```markdown
    /// ---
    /// name: Agent Browser
    /// description: A headless browser automation CLI
    /// read_when:
    ///   - Automating web interactions
    ///   - Extracting data from pages
    /// metadata: {"emoji":"🌐","requires":{"bins":["node"]}}
    /// allowed-tools: Bash(agent-browser:*)
    /// ---
    ///
    /// # Full instructions here...
    /// ```
    pub fn from_dir(path: &Path) -> Result<Self, SkillError> {
        let skill_md = path.join("SKILL.md");
        if !skill_md.exists() {
            return Err(SkillError::MissingSkillFile(
                path.display().to_string(),
            ));
        }

        let content = std::fs::read_to_string(&skill_md)?;
        Self::parse(&content)
    }

    /// 解析 SKILL.md 内容（frontmatter + body）
    pub fn parse(content: &str) -> Result<Self, SkillError> {
        let content = content.trim_start_matches('\u{FEFF}'); // strip BOM

        if !content.starts_with("---") {
            return Err(SkillError::Yaml("missing opening ---".into()));
        }

        // 找到第二个 ---
        let after_first = &content[3..];
        let end = after_first
            .find("---")
            .ok_or_else(|| SkillError::Yaml("missing closing ---".into()))?;

        let frontmatter_str = &after_first[..end];
        let body = after_first[end + 3..].trim();

        let fm: SkillFrontmatter = serde_yml::from_str(frontmatter_str)
            .map_err(|e| SkillError::Yaml(e.to_string()))?;

        if body.is_empty() {
            return Err(SkillError::Yaml("empty instructions body".into()));
        }

        Ok(Self {
            name: fm.name,
            description: fm.description,
            instructions: body.to_string(),
            read_when: fm.read_when,
            metadata: fm.metadata,
            allowed_tools: fm.allowed_tools,
        })
    }

    /// 生成注入上下文的摘要（简短）
    pub fn summary(&self) -> String {
        format!("- {}: {}", self.name, self.description)
    }

    /// 检查工具是否被 allowed-tools 允许
    pub fn is_tool_allowed(&self, tool_name: &str) -> bool {
        match &self.allowed_tools {
            Some(pattern) => {
                // 简单模式匹配：支持 "*" 通配符
                // "Bash(agent-browser:*)" → 匹配 "Bash" 工具
                // "*" → 匹配所有
                if pattern == "*" {
                    return true;
                }
                // 取 pattern 的第一段（工具名部分）
                let tool_pattern = pattern.split('(').next().unwrap_or(pattern);
                tool_name == tool_pattern || pattern.contains(tool_name)
            }
            None => true, // 无限制
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_skill_md() {
        let content = r#"---
name: Test Skill
description: A test skill for unit testing
read_when:
  - Testing things
  - Running unit tests
allowed-tools: Bash(test:*)
---

# Test Instructions

This is a comprehensive guide for testing.

1. Run `cargo test`
2. Check results
3. Fix any failures
"#;
        let skill = Skill::parse(content).unwrap();
        assert_eq!(skill.name, "Test Skill");
        assert_eq!(skill.description, "A test skill for unit testing");
        assert_eq!(skill.read_when, vec!["Testing things", "Running unit tests"]);
        assert!(skill.instructions.contains("Test Instructions"));
        assert!(skill.is_tool_allowed("Bash"));
        assert!(!skill.is_tool_allowed("write_file"));
    }

    #[test]
    fn test_no_allowed_tools_permits_all() {
        let content = r#"---
name: Open
description: No tool restrictions
read_when:
  - Anything
---

Do whatever.
"#;
        let skill = Skill::parse(content).unwrap();
        assert!(skill.is_tool_allowed("anything"));
    }

    #[test]
    fn test_wildcard_permits_all() {
        let content = r#"---
name: All
description: All tools
allowed-tools: "*"
---

Full access.
"#;
        let skill = Skill::parse(content).unwrap();
        assert!(skill.is_tool_allowed("anything"));
    }
}
