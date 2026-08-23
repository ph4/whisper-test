"""Metrics calculation utilities for ASR evaluation."""

import re
from typing import Any


def normalize_text(text: str) -> str:
    """Normalize text for WER/CER calculation.
    
    Args:
        text: Input text string.
        
    Returns:
        Normalized text (lowercase, no punctuation, extra spaces removed).
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    return text


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate between reference and hypothesis.
    
    WER = (S + D + I) / N
    where S=substitutions, D=deletions, I=insertions, N=total words in reference
    
    Uses Levenshtein distance algorithm without external dependencies.
    
    Args:
        reference: Ground truth text.
        hypothesis: Transcribed/hypothesized text.
        
    Returns:
        WER as a float (0.0 = perfect, 1.0 = all words wrong).
    """
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    
    # Levenshtein distance matrix
    m, n = len(ref_words), len(hyp_words)
    
    # Create distance matrix
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Deletion
                    dp[i][j - 1],      # Insertion
                    dp[i - 1][j - 1],  # Substitution
                )
    
    return dp[m][n] / len(ref_words)


def calculate_cer(reference: str, hypothesis: str) -> float:
    """Calculate Character Error Rate between reference and hypothesis.
    
    CER = (S + D + I) / N
    where S=substitutions, D=deletions, I=insertions, N=total chars in reference
    
    Args:
        reference: Ground truth text.
        hypothesis: Transcribed/hypothesized text.
        
    Returns:
        CER as a float (0.0 = perfect, 1.0 = all chars wrong).
    """
    ref_chars = normalize_text(reference)
    hyp_chars = normalize_text(hypothesis)
    
    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0
    
    # Levenshtein distance for characters
    m, n = len(ref_chars), len(hyp_chars)
    
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],
                    dp[i][j - 1],
                    dp[i - 1][j - 1],
                )
    
    return dp[m][n] / len(ref_chars)


def calculate_rtf(transcribe_time: float, audio_duration: float) -> float:
    """Calculate Real-Time Factor.
    
    RTF = transcribe_time / audio_duration
    
    RTF < 1.0 means faster than real-time (good).
    RTF > 1.0 means slower than real-time.
    
    Args:
        transcribe_time: Time taken for transcription in seconds.
        audio_duration: Duration of audio in seconds.
        
    Returns:
        RTF as float.
    """
    if audio_duration <= 0:
        return float("inf")
    return transcribe_time / audio_duration


def format_metrics(result: dict[str, Any], reference: str | None = None) -> str:
    """Format transcription results for console output.
    
    Args:
        result: Dictionary from transcriber.transcribe().
        reference: Optional ground truth text for WER/CER calculation.
        
    Returns:
        Formatted string for console display.
    """
    lines = [
        "=" * 60,
        f"Model: {result.get('model_name', 'Unknown')}",
        "=" * 60,
        f"Framework: {result.get('framework', 'N/A')}",
        f"Device: {result.get('device', 'N/A')}",
        "",
        "⏱️  Timing:",
        f"  Load time:      {result.get('load_time', 0):.3f}s",
        f"  Transcribe time:{result.get('transcribe_time', 0):.3f}s",
        f"  Audio duration: {result.get('duration', 0):.3f}s",
        f"  RTF:            {calculate_rtf(result.get('transcribe_time', 0), result.get('duration', 1)):.3f}",
        "",
        "💾 Memory:",
        f"  RAM peak:       {result.get('memory_peak_mb', 0):.1f} MB",
    ]
    
    if result.get('vram_peak_mb') is not None:
        lines.append(f"  VRAM peak:      {result.get('vram_peak_mb', 0):.1f} MB")
    
    if reference:
        wer = calculate_wer(reference, result.get('text', ''))
        cer = calculate_cer(reference, result.get('text', ''))
        lines.extend([
            "",
            "📊 Accuracy:",
            f"  WER:            {wer:.2%}",
            f"  CER:            {cer:.2%}",
        ])
    
    text_preview = result.get('text', '')[:100]
    if len(result.get('text', '')) > 100:
        text_preview += "..."
    
    lines.extend([
        "",
        "📝 Transcription (first 100 chars):",
        f"  \"{text_preview}\"",
        "=" * 60,
    ])
    
    return "\n".join(lines)


def calculate_word_count(text: str) -> int:
    """Count words in text.
    
    Args:
        text: Input text.
        
    Returns:
        Number of words.
    """
    return len(normalize_text(text).split())


def calculate_char_count(text: str) -> int:
    """Count non-whitespace characters in text.
    
    Args:
        text: Input text.
        
    Returns:
        Number of non-whitespace characters.
    """
    return len(normalize_text(text).replace(" ", ""))
