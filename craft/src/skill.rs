use std::path::{Path, PathBuf};

use yoakore::Skill;

/// Manages loading and selecting skills from a workspace directory.
pub struct SkillManager {
    #[allow(dead_code)]
    workspace: PathBuf,
    skills: Vec<Skill>,
}

impl SkillManager {
    /// Create a new SkillManager. Scans the workspace for SKILL.md files.
    pub fn new(workspace: impl Into<PathBuf>) -> Self {
        let workspace = workspace.into();
        let skills = Self::load_skills(&workspace);
        Self { workspace, skills }
    }

    /// Load all skills from the workspace directory.
    fn load_skills(workspace: &Path) -> Vec<Skill> {
        let skill_dir = workspace.join("skills");
        if !skill_dir.exists() {
            return Vec::new();
        }

        let mut skills = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&skill_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir()
                    && let Ok(skill) = Skill::from_dir(&path)
                {
                    skills.push(skill);
                }
            }
        }
        skills
    }

    /// Get all loaded skills.
    pub fn skills(&self) -> &[Skill] {
        &self.skills
    }

    /// Find skills matching a query using keyword matching.
    pub fn find_matching(&self, query: &str) -> Vec<&Skill> {
        let query_lower = query.to_lowercase();
        self.skills
            .iter()
            .filter(|s| {
                s.name.to_lowercase().contains(&query_lower)
                    || s.description.to_lowercase().contains(&query_lower)
                    || s.read_when
                        .iter()
                        .any(|t| t.to_lowercase().contains(&query_lower))
            })
            .collect()
    }

    /// Format matching skills for injection into the system prompt.
    pub fn format_for_prompt(&self, query: &str) -> Option<String> {
        let matching = self.find_matching(query);
        if matching.is_empty() {
            return None;
        }

        let formatted: Vec<String> = matching
            .iter()
            .map(|s| {
                format!(
                    "## Skill: {}\n{}\n\n{}",
                    s.name, s.description, s.instructions
                )
            })
            .collect();

        Some(format!(
            "<available-skills>\n{}\n</available-skills>",
            formatted.join("\n\n---\n\n")
        ))
    }
}
