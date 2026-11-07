"""
VoxSplit Multi-Format Transcript & Subtitle Exporter.
Serializes DiarizedUtterance streams into WebVTT (.vtt), SubRip (.srt),
Audacity Label Track (.txt), and readable script formats.
"""

from typing import List
from app.models.schemas import DiarizedUtterance, TranscriptionResponse


def format_timestamp_srt(seconds: float) -> str:
    """Formats seconds into SubRip timestamp: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Formats seconds into WebVTT timestamp: HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


class TranscriptExporter:
    """
    Exports structured transcription responses to standard captioning and editing formats.
    """

    @staticmethod
    def to_srt(utterances: List[DiarizedUtterance]) -> str:
        """
        Generates SubRip (.srt) subtitle content with speaker tags.
        """
        lines = []
        for idx, u in enumerate(utterances, start=1):
            start_str = format_timestamp_srt(u.start)
            end_str = format_timestamp_srt(u.end)
            lines.append(f"{idx}")
            lines.append(f"{start_str} --> {end_str}")
            lines.append(f"[{u.speaker}] {u.text}")
            lines.append("")  # Empty line separator
        return "\n".join(lines)

    @staticmethod
    def to_vtt(utterances: List[DiarizedUtterance]) -> str:
        """
        Generates WebVTT (.vtt) subtitle content with HTML5 <v Speaker> cue tags.
        """
        lines = ["WEBVTT", ""]
        for idx, u in enumerate(utterances, start=1):
            start_str = format_timestamp_vtt(u.start)
            end_str = format_timestamp_vtt(u.end)
            lines.append(f"{idx}")
            lines.append(f"{start_str} --> {end_str}")
            lines.append(f"<v {u.speaker}>{u.text}</v>")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def to_audacity_labels(utterances: List[DiarizedUtterance]) -> str:
        """
        Generates Audacity Label Track format (tab-delimited: start_sec \t end_sec \t label).
        Importable into Audacity, Praat, or Reaper for acoustic inspection.
        """
        lines = []
        for u in utterances:
            label = f"[{u.speaker}] {u.text}"
            lines.append(f"{u.start:.3f}\t{u.end:.3f}\t{label}")
        return "\n".join(lines)

    @staticmethod
    def to_plain_text_script(utterances: List[DiarizedUtterance]) -> str:
        """
        Generates a human-readable script dialogue transcript.
        """
        lines = []
        for u in utterances:
            start_m = int(u.start // 60)
            start_s = int(u.start % 60)
            end_m = int(u.end // 60)
            end_s = int(u.end % 60)
            time_tag = f"[{start_m:02d}:{start_s:02d} - {end_m:02d}:{end_s:02d}]"
            lines.append(f"{time_tag} {u.speaker}: {u.text}")
        return "\n".join(lines)

    @classmethod
    def export(cls, response: TranscriptionResponse, export_format: str) -> str:
        """
        Dispatches export to requested target format.
        Supported formats: 'srt', 'vtt', 'audacity', 'text', 'json'.
        """
        fmt = export_format.lower().strip()
        if fmt == "srt":
            return cls.to_srt(response.utterances)
        elif fmt == "vtt":
            return cls.to_vtt(response.utterances)
        elif fmt == "audacity":
            return cls.to_audacity_labels(response.utterances)
        elif fmt in ("text", "txt"):
            return cls.to_plain_text_script(response.utterances)
        elif fmt == "json":
            return response.model_dump_json(indent=2)
        else:
            raise ValueError(f"Unsupported export format '{export_format}'. Choose from: srt, vtt, audacity, text, json.")
