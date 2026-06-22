/// 文本分块：将长文本按固定长度切分，带重叠，按句子/段落边界切分
///
/// - `max_chars`: 每块最大字符数（约 2000 字符 ≈ 500 tokens）
/// - `overlap_chars`: 重叠字符数（约 400 字符 ≈ 100 tokens）
pub fn chunk_text(text: &str, max_chars: usize, overlap_chars: usize) -> Vec<String> {
    if text.len() <= max_chars {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let mut start = 0;

    while start < chars.len() {
        let end = (start + max_chars).min(chars.len());
        let slice: String = chars[start..end].iter().collect();

        // 尝试在末尾找到句子/段落边界（返回值为字符数）
        let adjusted = if end < chars.len() {
            find_boundary(&slice, max_chars).min(end - start)
        } else {
            end - start
        };

        let chunk: String = chars[start..start + adjusted].iter().collect();
        if !chunk.trim().is_empty() {
            chunks.push(chunk);
        }

        // 下一块从 overlap 位置开始（确保至少前进 1）
        let advance = adjusted.saturating_sub(overlap_chars).max(1);
        start += advance;
        if start >= chars.len() {
            break;
        }
    }

    chunks
}

/// 在文本中找到合适的切分边界（段落 > 句子 > 换行）
/// 返回值为字符数（非字节数），保证不会切到多字节 UTF-8 字符中间
fn find_boundary(text: &str, max_chars: usize) -> usize {
    let char_count = text.chars().count();
    let search_start_char = (max_chars * 2 / 3).min(char_count);

    // 将字符位置转换为字节偏移用于 str::find
    let search_start_byte: usize = text.chars().take(search_start_char).map(|c| c.len_utf8()).sum();

    // 优先找段落边界（双换行）
    if let Some(pos) = text[search_start_byte..].find("\n\n") {
        let boundary_byte = search_start_byte + pos + 2;
        return text[..boundary_byte].chars().count();
    }

    // 其次找句子边界（句号/问号/感叹号 + 换行或空格）
    let sentence_endings = [". ", ".\n", "! ", "!\n", "? ", "?\n", "。", "！", "？"];
    let mut best = max_chars;
    for ending in &sentence_endings {
        if let Some(pos) = text[search_start_byte..].find(ending) {
            let boundary_byte = search_start_byte + pos + ending.len();
            let boundary_char = text[..boundary_byte].chars().count();
            if boundary_char < best {
                best = boundary_char;
            }
        }
    }
    if best < max_chars {
        return best;
    }

    // 最后找换行
    if let Some(pos) = text[search_start_byte..].rfind('\n') {
        let boundary_byte = search_start_byte + pos + 1;
        return text[..boundary_byte].chars().count();
    }

    max_chars.min(char_count)
}

/// 估算文本的 token 数（粗略：1 token ≈ 4 字符 for English, ~2 字符 for CJK）
pub fn estimate_tokens(text: &str) -> i32 {
    let cjk_count = text.chars().filter(|c| is_cjk(*c)).count();
    let other_count = text.chars().count() - cjk_count;
    ((other_count / 4) + (cjk_count / 2)) as i32
}

fn is_cjk(c: char) -> bool {
    ('\u{4E00}'..='\u{9FFF}').contains(&c)
        || ('\u{3400}'..='\u{4DBF}').contains(&c)
        || ('\u{F900}'..='\u{FAFF}').contains(&c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_short_text_no_chunk() {
        let text = "Hello world";
        let chunks = chunk_text(text, 100, 20);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], text);
    }

    #[test]
    fn test_long_text_chunked() {
        let text = "This is a sentence. ".repeat(100); // ~2100 chars
        let chunks = chunk_text(&text, 500, 100);
        assert!(chunks.len() > 1);
        // 所有块拼起来应该覆盖原文
        let joined: String = chunks.join("");
        assert!(text.starts_with(&joined[..500]));
    }

    #[test]
    fn test_estimate_tokens() {
        let english = "Hello world, this is a test"; // 28 chars
        assert!(estimate_tokens(english) > 5);
        assert!(estimate_tokens(english) < 10);

        let chinese = "你好世界这是一个测试"; // 9 CJK chars
        assert!(estimate_tokens(chinese) > 3);
        assert!(estimate_tokens(chinese) < 8);
    }
}
