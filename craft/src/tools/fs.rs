use yoakore::prelude::*;

/// Register built-in file system tools into the given registry.
pub fn register_file_tools(registry: &mut ToolRegistry) {
    registry.register_async(
        ToolDefinition {
            name: "read_file".into(),
            description: "Read the contents of a file. Supports optional line ranges.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to the file" },
                    "offset": { "type": "integer", "description": "Start line (0-based)", "default": 0 },
                    "limit": { "type": "integer", "description": "Max lines to read", "default": 1000 }
                },
                "required": ["path"]
            }),
        },
        |args, _pm| Box::pin(async move {
            let path = args["path"].as_str().unwrap_or("").to_string();
            let offset = args["offset"].as_u64().unwrap_or(0) as usize;
            let limit = args["limit"].as_u64().unwrap_or(1000) as usize;
            let content = tokio::fs::read_to_string(&path).await
                .map_err(|e| format!("Failed to read {}: {}", path, e))?;
            let lines: Vec<&str> = content.lines().collect();
            let selected: Vec<String> = lines
                .iter()
                .skip(offset)
                .take(limit)
                .enumerate()
                .map(|(i, line)| format!("{}: {}", offset + i + 1, line))
                .collect();
            Ok(selected.join("\n"))
        }),
    );

    registry.register_async(
        ToolDefinition {
            name: "write_file".into(),
            description: "Write content to a file. Creates parent directories if needed.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to the file" },
                    "content": { "type": "string", "description": "Content to write" },
                    "append": { "type": "boolean", "description": "Append instead of overwrite", "default": false }
                },
                "required": ["path", "content"]
            }),
        },
        |args, _pm| Box::pin(async move {
            let path = args["path"].as_str().unwrap_or("").to_string();
            let content = args["content"].as_str().unwrap_or("").to_string();
            let append = args["append"].as_bool().unwrap_or(false);
            if let Some(parent) = std::path::Path::new(&path).parent() {
                tokio::fs::create_dir_all(parent).await
                    .map_err(|e| format!("Failed to create dirs: {}", e))?;
            }
            if append {
                use tokio::io::AsyncWriteExt;
                let mut file = tokio::fs::OpenOptions::new()
                    .create(true).append(true).open(&path).await
                    .map_err(|e| format!("Failed to open {}: {}", path, e))?;
                file.write_all(content.as_bytes()).await
                    .map_err(|e| format!("Failed to write: {}", e))?;
            } else {
                tokio::fs::write(&path, &content).await
                    .map_err(|e| format!("Failed to write {}: {}", path, e))?;
            }
            Ok(format!("Written {} bytes to {}", content.len(), path))
        }),
    );

    registry.register_async(
        ToolDefinition {
            name: "list_directory".into(),
            description: "List files and directories. Supports glob patterns.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Directory path", "default": "." },
                    "pattern": { "type": "string", "description": "Glob pattern (e.g. '*.rs')", "default": "*" }
                },
                "required": ["path"]
            }),
        },
        |args, _pm| Box::pin(async move {
            let path = args["path"].as_str().unwrap_or(".").to_string();
            let pattern = args["pattern"].as_str().unwrap_or("*").to_string();
            let glob_pattern = format!("{}/{}", path, pattern);
            let entries: Vec<String> = glob::glob(&glob_pattern)
                .map_err(|e| format!("Invalid glob: {}", e))?
                .filter_map(|e| e.ok())
                .map(|p| {
                    let meta = std::fs::metadata(&p);
                    let kind = meta.map(|m| if m.is_dir() { "[dir]" } else { "[file]" }).unwrap_or("[?]");
                    format!("{} {}", kind, p.display())
                })
                .collect();
            if entries.is_empty() {
                Ok("No matching entries found.".to_string())
            } else {
                Ok(entries.join("\n"))
            }
        }),
    );

    registry.register_async(
        ToolDefinition {
            name: "search_files".into(),
            description: "Search for text patterns in files. Returns matching lines with file paths.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Directory to search in", "default": "." },
                    "pattern": { "type": "string", "description": "Text or regex pattern to search" },
                    "glob": { "type": "string", "description": "File glob filter (e.g. '*.rs')", "default": "*" }
                },
                "required": ["pattern"]
            }),
        },
        |args, _pm| Box::pin(async move {
            let path = args["path"].as_str().unwrap_or(".").to_string();
            let pattern = args["pattern"].as_str().unwrap_or("").to_string();
            let glob_filter = args["glob"].as_str().unwrap_or("*").to_string();
            let re = regex::Regex::new(&pattern)
                .map_err(|e| format!("Invalid regex: {}", e))?;
            let glob_pattern = format!("{}/{}", path, glob_filter);
            let mut results = Vec::new();
            for entry in glob::glob(&glob_pattern).map_err(|e| format!("Invalid glob: {}", e))? {
                let path = match entry {
                    Ok(p) => p,
                    Err(_) => continue,
                };
                if !path.is_file() { continue; }
                let content = match std::fs::read_to_string(&path) {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                for (i, line) in content.lines().enumerate() {
                    if re.is_match(line) {
                        results.push(format!("{}:{}: {}", path.display(), i + 1, line));
                        if results.len() >= 50 {
                            return Ok(format!("{}\n... (truncated at 50 matches)", results.join("\n")));
                        }
                    }
                }
            }
            if results.is_empty() {
                Ok("No matches found.".to_string())
            } else {
                Ok(results.join("\n"))
            }
        }),
    );
}
