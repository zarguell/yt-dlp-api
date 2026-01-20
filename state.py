"""State management for task persistence with SQLite database."""

import datetime
import json
import logging
import sqlite3
import time
import uuid
from typing import Any

from fastapi import HTTPException

from models import JobType, Task
from utils import resolve_task_base_dir


class State:
    def __init__(self, db_file: str = "tasks.db", logger: logging.Logger | None = None):
        self.tasks: dict[str, Task] = {}
        self.db_file = db_file
        self.logger = logger or logging.getLogger("yt-dlp-api")
        self._init_db()
        self._load_tasks()

    def _init_db(self) -> None:
        self.logger.info("Initializing database db_file=%s", self.db_file)
        conn = sqlite3.connect(self.db_file)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                job_type TEXT NOT NULL,
                url TEXT NOT NULL,
                base_output_path TEXT NOT NULL,
                task_output_path TEXT NOT NULL,
                format TEXT NOT NULL,
                status TEXT NOT NULL,
                result TEXT,
                error TEXT,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.commit()
        conn.close()

    def _load_tasks(self) -> None:
        start = time.monotonic()
        try:
            conn = sqlite3.connect(self.db_file)
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, job_type, url, base_output_path, task_output_path,
                       format, status, result, error
                FROM tasks
                """
            )
            rows = cur.fetchall()
            for row in rows:
                (
                    task_id,
                    job_type,
                    url,
                    base_output_path,
                    task_output_path,
                    fmt,
                    status,
                    result_json,
                    error,
                ) = row
                result = json.loads(result_json) if result_json else None
                self.tasks[task_id] = Task(
                    id=task_id,
                    job_type=JobType(job_type),
                    url=url,
                    base_output_path=base_output_path,
                    task_output_path=task_output_path,
                    format=fmt,
                    status=status,
                    result=result,
                    error=error,
                )
            conn.close()
            self.logger.info(
                "Loaded tasks from database count=%d elapsed_ms=%d",
                len(rows),
                int((time.monotonic() - start) * 1000),
            )
        except Exception:
            self.logger.exception("Error loading tasks from database db_file=%s", self.db_file)

    def _save_task(self, task: Task) -> None:
        try:
            self.tasks[task.id] = task
            conn = sqlite3.connect(self.db_file)
            cur = conn.cursor()

            timestamp = datetime.datetime.now().isoformat()
            result_json = json.dumps(task.result) if task.result else None

            cur.execute(
                """
                INSERT OR REPLACE INTO tasks
                (id, job_type, url, base_output_path, task_output_path, format,
                 status, result, error, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task.id,
                    task.job_type.value,
                    task.url,
                    task.base_output_path,
                    task.task_output_path,
                    task.format,
                    task.status,
                    result_json,
                    task.error,
                    timestamp,
                ),
            )
            conn.commit()
            conn.close()
            self.logger.debug(
                "Saved task task_id=%s status=%s job_type=%s",
                task.id,
                task.status,
                task.job_type.value,
            )
        except Exception:
            self.logger.exception("Error saving task to database task_id=%s", task.id)

    def add_task(self, job_type: JobType, url: str, base_output_path: str, fmt: str) -> str:
        task_id = str(uuid.uuid4())
        base = resolve_task_base_dir(base_output_path)
        task_dir = (base / task_id).resolve(strict=False)

        if not task_dir.is_relative_to(base.resolve(strict=False)):
            self.logger.error(
                "Task dir containment check failed task_id=%s base=%s task_dir=%s",
                task_id,
                base,
                task_dir,
            )
            raise HTTPException(status_code=400, detail="Invalid task directory resolution.")

        task_dir.mkdir(parents=True, exist_ok=True)

        task = Task(
            id=task_id,
            job_type=job_type,
            url=url,
            base_output_path=str(base),
            task_output_path=str(task_dir),
            format=fmt,
            status="pending",
        )
        self._save_task(task)
        self.logger.info(
            "Created task task_id=%s job_type=%s base=%s fmt=%s url=%s",
            task_id,
            job_type.value,
            base,
            fmt,
            url,
        )
        return task_id

    def get_task(self, task_id: str) -> Task | None:
        return self.tasks.get(task_id)

    def update_task(
        self,
        task_id: str,
        status: str,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        task = self.tasks.get(task_id)
        if not task:
            self.logger.warning(
                "Attempted to update missing task task_id=%s status=%s", task_id, status
            )
            return

        task.status = status
        if result is not None:
            task.result = result
        if error is not None:
            task.error = error

        self._save_task(task)
        self.logger.info("Updated task task_id=%s status=%s", task_id, status)

    def list_tasks(self) -> list[Task]:
        return list(self.tasks.values())
