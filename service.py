"""
yt-dlp wrapper service module.

This module contains the YtDlpService class which wraps yt-dlp functionality
for video, audio, and subtitle downloads.
"""

import logging
import time
from pathlib import Path
from typing import Any, Sequence, cast

import yt_dlp

from models import EnglishMode, SubtitleFormat, SubtitlePreference
from utils import ensure_dir

logger = logging.getLogger("yt-dlp-api")


class YtDlpService:
    """
    Wrapper around yt-dlp library for downloading media content.

    All methods are static and thread-safe, using yt-dlp's context managers
    for proper resource management.
    """

    @staticmethod
    def get_info(url: str, quiet: bool = False) -> dict[str, Any]:
        """Extract video information without downloading."""
        opts = {"quiet": quiet, "no_warnings": quiet, "skip_download": True}
        logger.debug("yt-dlp get_info url=%s quiet=%s", url, quiet)
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return cast("dict[str, Any]", ydl.sanitize_info(info))

    @staticmethod
    def list_formats(url: str) -> list[dict[str, Any]]:
        """List available formats for a video."""
        info = YtDlpService.get_info(url=url, quiet=True)
        return info.get("formats", []) if info else []

    @staticmethod
    def download_video(
        url: str,
        output_path: str,
        fmt: str,
        quiet: bool,
        cookie_file: str | None = None,
    ) -> dict[str, Any]:
        """Download video in specified format."""
        ensure_dir(output_path)
        outtmpl = str(Path(output_path) / "%(title).180s.%(ext)s")
        ydl_opts = {
            "outtmpl": outtmpl,
            "quiet": quiet,
            "no_warnings": quiet,
            "format": fmt,
            "no_abort_on_error": True,
            "sleep_interval": 10,
            "sleep_subtitles": 10,
        }

        # Add cookies if provided (already validated by resolve_cookie_file)
        if cookie_file:
            ydl_opts["cookiefile"] = cookie_file
            logger.info("Using cookies file path=%s", cookie_file)

        logger.info(
            "yt-dlp download_video start url=%s output_path=%s fmt=%s quiet=%s cookie_file=%s",
            url,
            output_path,
            fmt,
            quiet,
            cookie_file,
        )
        start = time.monotonic()
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.info("yt-dlp download_video done url=%s elapsed_ms=%d", url, elapsed_ms)
            return cast("dict[str, Any]", ydl.sanitize_info(info))

    @staticmethod
    def download_audio(
        url: str,
        output_path: str,
        audio_format: str,
        audio_quality: str | None,
        quiet: bool,
        cookie_file: str | None = None,
    ) -> dict[str, Any]:
        """Download audio in specified format."""
        ensure_dir(output_path)
        outtmpl = str(Path(output_path) / "%(title).180s.%(ext)s")
        ydl_opts: dict[str, Any] = {
            "outtmpl": outtmpl,
            "quiet": quiet,
            "no_warnings": quiet,
            "format": "bestaudio/best",
            "extractaudio": True,
            "audioformat": audio_format,
            "no_abort_on_error": True,
            "sleep_interval": 10,
            "sleep_subtitles": 10,
        }
        if audio_quality is not None:
            ydl_opts["audioquality"] = audio_quality

        # Add cookies if provided (already validated by resolve_cookie_file)
        if cookie_file:
            ydl_opts["cookiefile"] = cookie_file
            logger.info("Using cookies file path=%s", cookie_file)

        logger.info(
            "yt-dlp download_audio start url=%s output_path=%s audio_format=%s audio_quality=%s quiet=%s cookie_file=%s",
            url,
            output_path,
            audio_format,
            audio_quality,
            quiet,
            cookie_file,
        )
        start = time.monotonic()
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.info("yt-dlp download_audio done url=%s elapsed_ms=%d", url, elapsed_ms)
            return cast("dict[str, Any]", ydl.sanitize_info(info))

    @staticmethod
    def download_subtitles(
        url: str,
        output_path: str,
        languages: Sequence[str],
        write_manual: bool,
        write_automatic: bool,
        convert_to: str | None,
        quiet: bool,
        cookie_file: str | None = None,
    ) -> dict[str, Any]:
        """
        Download subtitles with support for partial success tracking.

        Returns a result dictionary with:
        - 'success': True if all requested subtitles were downloaded
        - 'downloaded': List of successfully downloaded subtitle files
        - 'failed': List of subtitle downloads that failed (empty if full success)
        - 'info': yt-dlp info dict (may be partial if download failed)
        - 'error': Error message if download failed
        """
        ensure_dir(output_path)
        outtmpl = str(Path(output_path) / "%(title).180s.%(ext)s")
        ydl_opts: dict[str, Any] = {
            "outtmpl": outtmpl,
            "quiet": quiet,
            "no_warnings": quiet,
            "skip_download": True,
            "subtitleslangs": list(languages),
            "no_abort_on_error": True,
            "sleep_interval": 10,
            "sleep_subtitles": 10,
            # Workaround: avoid WEB player client for extraction
            "extractor_args": {
                "youtube": {
                    "player_client": ["default", "-web"],
                }
            },
        }
        if write_manual:
            ydl_opts["writesubtitles"] = True
        if write_automatic:
            ydl_opts["writeautomaticsub"] = True
        if convert_to:
            ydl_opts["convertsubtitles"] = convert_to

        # Add cookies if provided (already validated by resolve_cookie_file)
        if cookie_file:
            ydl_opts["cookiefile"] = cookie_file
            logger.info("Using cookies file path=%s", cookie_file)

        logger.info(
            "yt-dlp download_subtitles start url=%s output_path=%s languages=%s manual=%s auto=%s convert_to=%s quiet=%s cookie_file=%s",
            url,
            output_path,
            list(languages),
            write_manual,
            write_automatic,
            convert_to,
            quiet,
            cookie_file,
        )
        start = time.monotonic()

        # Track files before download
        output_dir = Path(output_path)
        files_before = set(output_dir.glob("*")) if output_dir.exists() else set()

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                elapsed_ms = int((time.monotonic() - start) * 1000)
                sanitized_info = ydl.sanitize_info(info)

                # Check what files were actually created
                files_after = set(output_dir.glob("*")) if output_dir.exists() else set()
                new_files = files_after - files_before

                # Extract subtitle information from the result
                subtitles_data = {}
                if info and "subtitles" in info:
                    subtitles_data = info["subtitles"]
                if info and "automatic_captions" in info:
                    if write_automatic:
                        subtitles_data.update(info["automatic_captions"])

                # Determine which subtitles were actually downloaded
                downloaded_files = []
                for f in new_files:
                    if f.is_file():
                        downloaded_files.append(
                            {
                                "name": f.name,
                                "size_bytes": f.stat().st_size,
                                "path": str(f),
                            }
                        )

                # Check if we got the expected subtitles
                # Extract available subtitle languages from info
                requested_count = len(languages)
                successful_downloads = len(downloaded_files)

                logger.info(
                    "yt-dlp download_subtitles done url=%s elapsed_ms=%d downloaded=%d requested=%d",
                    url,
                    elapsed_ms,
                    successful_downloads,
                    requested_count,
                )

                return {
                    "success": successful_downloads > 0,
                    "downloaded": downloaded_files,
                    "failed": [] if successful_downloads > 0 else ["All subtitle downloads failed"],
                    "info": sanitized_info,
                    "partial": successful_downloads > 0 and successful_downloads < requested_count,
                }

        except Exception as e:
            # Partial success: some files may have been created before error
            files_after = set(output_dir.glob("*")) if output_dir.exists() else set()
            new_files = files_after - files_before

            downloaded_files = []
            for f in new_files:
                if f.is_file():
                    downloaded_files.append(
                        {
                            "name": f.name,
                            "size_bytes": f.stat().st_size,
                            "path": str(f),
                        }
                    )

            elapsed_ms = int((time.monotonic() - start) * 1000)
            error_msg = str(e)

            # Check if it's a retryable error
            is_429 = "429" in error_msg or "too many requests" in error_msg.lower()

            logger.warning(
                "yt-dlp download_subtitles failed url=%s elapsed_ms=%d downloaded_before_error=%d error=%s",
                url,
                elapsed_ms,
                len(downloaded_files),
                error_msg[:200],
            )

            return {
                "success": False,
                "downloaded": downloaded_files,
                "failed": [error_msg],
                "info": None,
                "error": error_msg,
                "partial": len(downloaded_files) > 0,
                "is_retryable": is_429,
            }

    @staticmethod
    def _select_best_subtitle_language(
        info: dict[str, Any],
        english_rank: list[str],
        prefer: SubtitlePreference,
    ) -> str | None:
        """
        Select the best available subtitle language based on ranking and preference.

        Args:
            info: Video info dict from yt-dlp (must contain 'subtitles' and 'automatic_captions')
            english_rank: Ordered list of language patterns (supports regex)
            prefer: Whether to prefer manual, automatic, or both

        Returns:
            Selected language tag (e.g., 'en', 'en-US') or None if no match found
        """
        import re

        manual_subs = info.get("subtitles", {})
        auto_subs = info.get("automatic_captions", {})

        # Build available language sets
        manual_langs = set(manual_subs.keys())
        auto_langs = set(auto_subs.keys())

        logger.debug(
            "Available subtitles manual=%s auto=%s",
            sorted(manual_langs),
            sorted(auto_langs),
        )

        # Try each pattern in ranking order
        for pattern in english_rank:
            # Check for exact match first (faster)
            if pattern in manual_langs and prefer != SubtitlePreference.auto_only:
                logger.info("Selected manual subtitle exact_match=%s", pattern)
                return pattern
            if pattern in auto_langs and prefer != SubtitlePreference.manual_only:
                logger.info("Selected automatic caption exact_match=%s", pattern)
                return pattern

            # Try regex match
            escaped = re.escape(pattern).replace(r"\*", ".*")
            regex = re.compile(f"^{escaped}$")
            manual_matches = [lang for lang in manual_langs if regex.match(lang)]
            auto_matches = [lang for lang in auto_langs if regex.match(lang)]

            # Prefer manual over auto based on preference
            if prefer == SubtitlePreference.manual_only:
                if manual_matches:
                    selected = manual_matches[0]
                    logger.info("Selected manual subtitle regex=%s match=%s", pattern, selected)
                    return selected
            elif prefer == SubtitlePreference.auto_only:
                if auto_matches:
                    selected = auto_matches[0]
                    logger.info("Selected automatic caption regex=%s match=%s", pattern, selected)
                    return selected
            else:  # manual_then_auto
                if manual_matches:
                    selected = manual_matches[0]
                    logger.info("Selected manual subtitle regex=%s match=%s", pattern, selected)
                    return selected
                if auto_matches:
                    selected = auto_matches[0]
                    logger.info("Selected automatic caption regex=%s match=%s", pattern, selected)
                    return selected

        logger.warning("No matching subtitle language found for patterns=%s", english_rank)
        return None

    @staticmethod
    def _get_all_english_languages(info: dict[str, Any], prefer: SubtitlePreference) -> list[str]:
        """
        Get all English language variants available.

        Args:
            info: Video info dict from yt-dlp
            prefer: Manual vs automatic preference

        Returns:
            List of English language tags
        """
        import re

        manual_subs = info.get("subtitles", {})
        auto_subs = info.get("automatic_captions", {})

        # Match English variants (en, en-US, en-GB, etc.)
        english_regex = re.compile(r"^en(-[A-Z]{2})?$")

        langs = set()
        if prefer != SubtitlePreference.auto_only:
            langs.update(lang for lang in manual_subs.keys() if english_regex.match(lang))
        if prefer != SubtitlePreference.manual_only:
            langs.update(lang for lang in auto_subs.keys() if english_regex.match(lang))

        return sorted(langs)

    @staticmethod
    def download_subtitles_v2(
        url: str,
        output_path: str,
        english_mode: EnglishMode,
        languages: list[str],
        prefer: SubtitlePreference,
        formats: SubtitleFormat,
        english_rank: list[str],
        quiet: bool,
        cookie_file: str | None = None,
    ) -> dict[str, Any]:
        """
        Enhanced subtitle download with policy-based language selection.

        Algorithm:
        1. Extract video info to inspect available subtitles
        2. Select language(s) based on english_mode policy
        3. Download with optimal yt-dlp options for format preference
        4. Return downloaded files with metadata

        Returns:
            Dict with:
            - 'success': bool
            - 'downloaded': list of file info dicts
            - 'selected_languages': list of language tags that were selected
            - 'info': full video info dict
            - 'error': error message if failed
        """
        # Get video info first
        logger.info("Extracting video info for subtitle selection url=%s", url)
        info_opts = {
            "quiet": quiet,
            "no_warnings": quiet,
            "skip_download": True,
        }
        if cookie_file:
            info_opts["cookiefile"] = cookie_file

        try:
            with yt_dlp.YoutubeDL(info_opts) as ydl:
                info = ydl.extract_info(url, download=False)
        except Exception as e:
            logger.error("Failed to extract video info error=%s", str(e))
            return {
                "success": False,
                "downloaded": [],
                "selected_languages": [],
                "info": None,
                "error": f"Failed to extract video info: {e}",
            }

        # Step 2: Select languages based on policy
        selected_languages: list[str]

        if english_mode == EnglishMode.explicit:
            if not languages:
                return {
                    "success": False,
                    "downloaded": [],
                    "selected_languages": [],
                    "info": ydl.sanitize_info(info),
                    "error": "english_mode='explicit' requires non-empty languages list",
                }
            selected_languages = languages
            logger.info("Using explicit languages=%s", selected_languages)

        elif english_mode == EnglishMode.best_one:
            # Select single best language
            lang = YtDlpService._select_best_subtitle_language(info, english_rank, prefer)
            if not lang:
                return {
                    "success": False,
                    "downloaded": [],
                    "selected_languages": [],
                    "info": ydl.sanitize_info(info),
                    "error": f"No English subtitles found (prefer={prefer.value}, tried patterns={english_rank})",
                }
            selected_languages = [lang]
            logger.info("Selected best_one language=%s", lang)

        else:  # all_english
            selected_languages = YtDlpService._get_all_english_languages(info, prefer)
            if not selected_languages:
                return {
                    "success": False,
                    "downloaded": [],
                    "selected_languages": [],
                    "info": ydl.sanitize_info(info),
                    "error": f"No English subtitles found (prefer={prefer.value})",
                }
            logger.info("Selected all_english languages=%s", selected_languages)

        # Step 3: Configure yt-dlp options for format preference
        ensure_dir(output_path)
        outtmpl = str(Path(output_path) / "%(title).180s.%(ext)s")

        # Configure format handling
        if formats == SubtitleFormat.vtt:
            # Prefer VTT, no conversion
            subtitles_format = "vtt/best"
            convert_subtitles = None
        elif formats == SubtitleFormat.srt:
            # Prefer SRT, convert if needed
            subtitles_format = "srt/best"
            convert_subtitles = "srt"
        else:  # both
            # Get VTT as primary, convert to SRT
            subtitles_format = "vtt/best"
            convert_subtitles = "srt"

        # Configure manual vs automatic
        write_manual = prefer != SubtitlePreference.auto_only
        write_auto = prefer != SubtitlePreference.manual_only

        ydl_opts: dict[str, Any] = {
            "outtmpl": outtmpl,
            "quiet": quiet,
            "no_warnings": quiet,
            "skip_download": True,
            "subtitleslangs": selected_languages,
            "subtitlesformat": subtitles_format,
            "no_abort_on_error": True,
            "sleep_interval": 10,
            "sleep_subtitles": 10,
            "extractor_args": {
                "youtube": {
                    "player_client": ["default", "-web"],
                }
            },
        }

        if write_manual:
            ydl_opts["writesubtitles"] = True
        if write_auto:
            ydl_opts["writeautomaticsub"] = True
        if convert_subtitles:
            ydl_opts["convertsubtitles"] = convert_subtitles
        if cookie_file:
            ydl_opts["cookiefile"] = cookie_file

        logger.info(
            "yt-dlp download_subtitles_v2 start url=%s output_path=%s languages=%s format=%s convert=%s manual=%s auto=%s",
            url,
            output_path,
            selected_languages,
            subtitles_format,
            convert_subtitles,
            write_manual,
            write_auto,
        )

        # Step 4: Download subtitles
        start = time.monotonic()
        output_dir = Path(output_path)
        files_before = set(output_dir.glob("*")) if output_dir.exists() else set()

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download(url)
                elapsed_ms = int((time.monotonic() - start) * 1000)

            # Check what files were created
            files_after = set(output_dir.glob("*")) if output_dir.exists() else set()
            new_files = files_after - files_before

            downloaded_files = []
            for f in new_files:
                if f.is_file():
                    downloaded_files.append(
                        {
                            "name": f.name,
                            "size_bytes": f.stat().st_size,
                            "path": str(f),
                        }
                    )

            logger.info(
                "yt-dlp download_subtitles_v2 done url=%s elapsed_ms=%d downloaded=%d files",
                url,
                elapsed_ms,
                len(downloaded_files),
            )

            return {
                "success": len(downloaded_files) > 0,
                "downloaded": downloaded_files,
                "selected_languages": selected_languages,
                "info": ydl.sanitize_info(info),
                "error": None if downloaded_files else "No subtitle files were downloaded",
            }

        except Exception as e:
            # Partial success: some files may have been created
            files_after = set(output_dir.glob("*")) if output_dir.exists() else set()
            new_files = files_after - files_before

            downloaded_files = []
            for f in new_files:
                if f.is_file():
                    downloaded_files.append(
                        {
                            "name": f.name,
                            "size_bytes": f.stat().st_size,
                            "path": str(f),
                        }
                    )

            error_msg = str(e)
            logger.warning(
                "yt-dlp download_subtitles_v2 failed url=%s downloaded_before_error=%d error=%s",
                url,
                len(downloaded_files),
                error_msg[:200],
            )

            return {
                "success": False,
                "downloaded": downloaded_files,
                "selected_languages": selected_languages,
                "info": ydl.sanitize_info(info),
                "error": error_msg,
            }
