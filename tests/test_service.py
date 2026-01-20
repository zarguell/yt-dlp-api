"""
Unit tests for YtDlpService class.

Tests are organized by method and use mocking to avoid actual network calls to yt-dlp.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from models import EnglishMode, SubtitleFormat, SubtitlePreference
from service import YtDlpService

logger = logging.getLogger("yt-dlp-api")


class TestYtDlpServiceGetInfo:
    """Tests for YtDlpService.get_info() static method."""

    @patch("service.yt_dlp.YoutubeDL")
    def test_get_info_returns_sanitized_info(self, mock_ytdl_class):
        """Test that get_info returns sanitized info dict."""
        # Setup mock
        mock_ytdl_instance = MagicMock()
        mock_ytdl_class.return_value.__enter__.return_value = mock_ytdl_instance

        expected_info = {"id": "test123", "title": "Test Video"}
        mock_ytdl_instance.extract_info.return_value = expected_info
        mock_ytdl_instance.sanitize_info.return_value = expected_info

        # Execute
        result = YtDlpService.get_info(url="https://youtube.com/watch?v=test123", quiet=True)

        # Verify
        assert result == expected_info
        mock_ytdl_instance.extract_info.assert_called_once_with(
            "https://youtube.com/watch?v=test123", download=False
        )
        mock_ytdl_instance.sanitize_info.assert_called_once_with(expected_info)

    @patch("service.yt_dlp.YoutubeDL")
    def test_get_info_passes_quiet_option(self, mock_ytdl_class):
        """Test that get_info passes quiet parameter correctly."""
        mock_ytdl_instance = MagicMock()
        mock_ytdl_class.return_value.__enter__.return_value = mock_ytdl_instance

        mock_ytdl_instance.extract_info.return_value = {}
        mock_ytdl_instance.sanitize_info.return_value = {}

        # Test with quiet=True
        YtDlpService.get_info(url="https://youtube.com/watch?v=test", quiet=True)
        args, kwargs = mock_ytdl_class.call_args
        assert args[0]["quiet"] is True
        assert args[0]["no_warnings"] is True

        # Test with quiet=False
        YtDlpService.get_info(url="https://youtube.com/watch?v=test", quiet=False)
        args, kwargs = mock_ytdl_class.call_args
        assert args[0]["quiet"] is False
        assert args[0]["no_warnings"] is False


class TestYtDlpServiceListFormats:
    """Tests for YtDlpService.list_formats() static method."""

    @patch("service.YtDlpService.get_info")
    def test_list_formats_extracts_formats_list(self, mock_get_info):
        """Test that list_formats returns formats from get_info."""
        # Setup mock
        mock_info = {
            "formats": [
                {"format_id": "137", "ext": "mp4", "height": 1080},
                {"format_id": "140", "ext": "m4a"},
            ]
        }
        mock_get_info.return_value = mock_info

        # Execute
        result = YtDlpService.list_formats(url="https://youtube.com/watch?v=test")

        # Verify
        assert result == mock_info["formats"]
        mock_get_info.assert_called_once_with(url="https://youtube.com/watch?v=test", quiet=True)

    @patch("service.YtDlpService.get_info")
    def test_list_formats_returns_empty_list_when_no_formats(self, mock_get_info):
        """Test that list_formats returns empty list when no formats available."""
        # Setup mock - info with no formats key
        mock_get_info.return_value = {}

        # Execute
        result = YtDlpService.list_formats(url="https://youtube.com/watch?v=test")

        # Verify
        assert result == []

    @patch("service.YtDlpService.get_info")
    def test_list_formats_returns_empty_list_when_info_is_none(self, mock_get_info):
        """Test that list_formats returns empty list when get_info returns None."""
        # Setup mock
        mock_get_info.return_value = None

        # Execute
        result = YtDlpService.list_formats(url="https://youtube.com/watch?v=test")

        # Verify
        assert result == []


class TestYtDlpServiceSelectBestSubtitleLanguage:
    """Tests for YtDlpService._select_best_subtitle_language() static method."""

    def test_select_best_exact_match_manual(self):
        """Test exact match selection with manual subtitles."""
        info = {
            "subtitles": {"en": [{"data": "manual subs"}], "es": [{"data": "spanish"}]},
            "automatic_captions": {},
        }
        english_rank = ["en"]
        prefer = SubtitlePreference.manual_then_auto

        result = YtDlpService._select_best_subtitle_language(info, english_rank, prefer)

        assert result == "en"

    def test_select_best_exact_match_auto(self):
        """Test exact match selection with automatic captions."""
        info = {
            "subtitles": {},
            "automatic_captions": {"en": [{"data": "auto subs"}]},
        }
        english_rank = ["en"]
        prefer = SubtitlePreference.auto_only

        result = YtDlpService._select_best_subtitle_language(info, english_rank, prefer)

        assert result == "en"

    def test_select_best_regex_match(self):
        """Test regex match selection for language variants."""
        info = {
            "subtitles": {},
            "automatic_captions": {
                "en-US": [{"data": "en-US auto"}],
                "en-GB": [{"data": "en-GB auto"}],
            },
        }
        english_rank = ["en-*"]
        prefer = SubtitlePreference.auto_only

        result = YtDlpService._select_best_subtitle_language(info, english_rank, prefer)

        assert result in ["en-US", "en-GB"]  # Either match is acceptable

    def test_select_best_preference_manual_only(self):
        """Test manual_only preference ignores auto captions."""
        info = {
            "subtitles": {},
            "automatic_captions": {"en": [{"data": "auto subs"}]},
        }
        english_rank = ["en"]
        prefer = SubtitlePreference.manual_only

        result = YtDlpService._select_best_subtitle_language(info, english_rank, prefer)

        assert result is None  # No manual subtitles available

    def test_select_best_preference_auto_only(self):
        """Test auto_only preference ignores manual subtitles."""
        info = {
            "subtitles": {"en": [{"data": "manual subs"}]},
            "automatic_captions": {},
        }
        english_rank = ["en"]
        prefer = SubtitlePreference.auto_only

        result = YtDlpService._select_best_subtitle_language(info, english_rank, prefer)

        assert result is None  # No auto captions available

    def test_select_best_no_match_found(self):
        """Test when no matching subtitle language is found."""
        info = {
            "subtitles": {"es": [{"data": "spanish"}]},
            "automatic_captions": {"fr": [{"data": "french"}]},
        }
        english_rank = ["en", "en-US"]
        prefer = SubtitlePreference.manual_then_auto

        result = YtDlpService._select_best_subtitle_language(info, english_rank, prefer)

        assert result is None


class TestYtDlpServiceGetAllEnglishLanguages:
    """Tests for YtDlpService._get_all_english_languages() static method."""

    def test_get_all_english_variants(self):
        """Test extracting all English language variants."""
        info = {
            "subtitles": {
                "en": [{"data": "en"}],
                "en-US": [{"data": "en-US"}],
                "en-GB": [{"data": "en-GB"}],
            },
            "automatic_captions": {
                "en-AU": [{"data": "en-AU"}],
            },
        }
        prefer = SubtitlePreference.manual_then_auto

        result = YtDlpService._get_all_english_languages(info, prefer)

        assert sorted(result) == ["en", "en-AU", "en-GB", "en-US"]

    def test_get_all_english_manual_only(self):
        """Test with manual_only preference."""
        info = {
            "subtitles": {
                "en": [{"data": "en"}],
                "en-US": [{"data": "en-US"}],
            },
            "automatic_captions": {
                "en-AU": [{"data": "en-AU"}],
            },
        }
        prefer = SubtitlePreference.manual_only

        result = YtDlpService._get_all_english_languages(info, prefer)

        assert sorted(result) == ["en", "en-US"]
        assert "en-AU" not in result  # Auto captions excluded

    def test_get_all_english_auto_only(self):
        """Test with auto_only preference."""
        info = {
            "subtitles": {
                "en": [{"data": "en"}],
            },
            "automatic_captions": {
                "en-AU": [{"data": "en-AU"}],
            },
        }
        prefer = SubtitlePreference.auto_only

        result = YtDlpService._get_all_english_languages(info, prefer)

        assert result == ["en-AU"]
        assert "en" not in result  # Manual subtitles excluded

    def test_get_all_english_filters_non_english(self):
        """Test that non-English languages are filtered out."""
        info = {
            "subtitles": {
                "en": [{"data": "english"}],
                "es": [{"data": "spanish"}],
                "fr": [{"data": "french"}],
            },
            "automatic_captions": {},
        }
        prefer = SubtitlePreference.manual_then_auto

        result = YtDlpService._get_all_english_languages(info, prefer)

        assert result == ["en"]
        assert "es" not in result
        assert "fr" not in result

    def test_get_all_english_no_matches(self):
        """Test when no English subtitles are available."""
        info = {
            "subtitles": {
                "es": [{"data": "spanish"}],
                "fr": [{"data": "french"}],
            },
            "automatic_captions": {},
        }
        prefer = SubtitlePreference.manual_then_auto

        result = YtDlpService._get_all_english_languages(info, prefer)

        assert result == []
