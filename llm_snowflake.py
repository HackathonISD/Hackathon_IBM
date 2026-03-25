"""Snowflake Cortex LLM client.

Provides a high-level interface to Snowflake Cortex COMPLETE for text and
multimodal inference, including internal-stage image upload utilities.
"""

import os
import json
import re
import tempfile
from dotenv import load_dotenv
import snowflake.connector
from typing import Optional


class SnowflakeCortexClient:
    """Client for invoking Snowflake Cortex AI models.

    Manages a persistent Snowflake connection and exposes methods for
    text-only and multimodal (stage-hosted image) completions via the
    ``SNOWFLAKE.CORTEX.COMPLETE`` SQL function.
    """

    def __init__(
        self,
        account: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        warehouse: Optional[str] = None,
        database: Optional[str] = None,
        schema: Optional[str] = None,
        role: Optional[str] = None,
    ):
        """Initialize the Snowflake Cortex client.

        All connection parameters fall back to their corresponding
        ``SNOWFLAKE_*`` environment variables when not supplied.

        Args:
            account: Snowflake account identifier.
            user: Snowflake username.
            password: Snowflake password.
            warehouse: Snowflake warehouse name.
            database: Snowflake database name.
            schema: Snowflake schema name.
            role: Snowflake role name.
        """
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        load_dotenv(env_path)
        self.account = account or os.getenv("SNOWFLAKE_ACCOUNT", "")
        self.user = user or os.getenv("SNOWFLAKE_USER", "")
        self.password = password or os.getenv("SNOWFLAKE_PASSWORD", "")
        self.warehouse = warehouse or os.getenv("SNOWFLAKE_WAREHOUSE", "")
        self.database = database or os.getenv("SNOWFLAKE_DATABASE", "")
        self.schema = schema or os.getenv("SNOWFLAKE_SCHEMA", "")
        self.role = role or os.getenv("SNOWFLAKE_ROLE", "")
        self.connection = None
        self._stage_created: set[str] = set()

    # ------------------------------------------------------------------ #
    #  Connection management
    # ------------------------------------------------------------------ #

    def _connect(self):
        """Establish a connection to Snowflake if not already connected."""
        if self.connection is None:
            connect_args = dict(
                account=self.account,
                user=self.user,
                password=self.password,
                warehouse=self.warehouse,
                database=self.database,
                schema=self.schema,
            )
            if self.role:
                connect_args["role"] = self.role
            self.connection = snowflake.connector.connect(**connect_args)
            session_cursor = self.connection.cursor()
            try:
                if self.role:
                    session_cursor.execute(f"USE ROLE {self.role}")
                if self.warehouse:
                    session_cursor.execute(f"USE WAREHOUSE {self.warehouse}")
                if self.database:
                    session_cursor.execute(f"USE DATABASE {self.database}")
                if self.schema:
                    session_cursor.execute(f"USE SCHEMA {self.schema}")
            finally:
                session_cursor.close()

    def close(self):
        """Close the active Snowflake connection, if any."""
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    # ------------------------------------------------------------------ #
    #  Text completion
    # ------------------------------------------------------------------ #

    def call(
        self,
        model: str,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        temperature: float = 0.0,
        image_urls: Optional[list[str]] = None,
    ) -> str:
        """Generate a completion via Snowflake Cortex COMPLETE.

        Uses the structured 3-argument form ``COMPLETE(model, messages, options)``
        for explicit temperature control. When *image_urls* are supplied the
        request is delegated to the multimodal path.

        Args:
            model: Model identifier (e.g. ``'openai-gpt-4.1'``).
            prompt: Direct prompt text. Ignored when *user_prompt* is set.
            system_prompt: Optional system-level instructions.
            user_prompt: User message text.
            temperature: Sampling temperature in ``[0, 2]``.
            image_urls: Stage references (e.g. ``['@stage/img.png']``) for
                multimodal requests.

        Returns:
            Generated response string.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_text = user_prompt or prompt

        if user_text and image_urls:
            return self._call_with_to_file(
                model=model,
                prompt=user_text,
                image_urls=image_urls,
                system_prompt=system_prompt,
                temperature=temperature,
            )

        if user_text:
            messages.append({"role": "user", "content": user_text})

        if not messages:
            return "Error: Either 'prompt' or both 'system_prompt'/'user_prompt' must be provided."

        options = {
            "temperature": temperature,
            "max_tokens": 8192,
        }

        cursor = None
        try:
            self._connect()
            cursor = self.connection.cursor()

            sql_query = "SELECT SNOWFLAKE.CORTEX.COMPLETE(%s, PARSE_JSON(%s), PARSE_JSON(%s)) AS response"
            cursor.execute(
                sql_query, (model, json.dumps(messages), json.dumps(options))
            )
            result = cursor.fetchone()

            if result:
                raw = result[0]
                try:
                    obj = json.loads(raw) if isinstance(raw, str) else raw
                    if isinstance(obj, dict) and "choices" in obj:
                        return obj["choices"][0].get("messages", raw)
                except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                    pass
                return raw
            return "No response received from Cortex."

        except snowflake.connector.errors.DatabaseError as e:
            return f"Database Error: {str(e)}"
        except Exception as e:
            return f"An error occurred: {str(e)}"
        finally:
            if cursor is not None:
                cursor.close()

    # ------------------------------------------------------------------ #
    #  Multimodal completion
    # ------------------------------------------------------------------ #

    def _call_with_to_file(
        self,
        model: str,
        prompt: str,
        image_urls: list[str],
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> str:
        """Route a multimodal request through ``TO_FILE()``.

        Stage references cannot be embedded inside ``PARSE_JSON()``; they
        require ``TO_FILE()`` evaluation at query time.  Only the first
        valid stage reference is used (simple 3-arg form limitation).

        Args:
            model: Model identifier.
            prompt: User prompt text.
            image_urls: Stage URLs (``@stage/file``).
            system_prompt: Optional system instructions (prepended to prompt).
            temperature: Sampling temperature.

        Returns:
            Generated response string.
        """
        refs: list[tuple[str, str]] = []
        for url in image_urls:
            m = re.match(r"^@([A-Za-z_][A-Za-z0-9_.]*)/(.+)$", url)
            if m:
                refs.append((m.group(1), m.group(2)))
        if not refs:
            return self.call(
                model=model,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
            )

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        return self.call_with_image(
            model=model,
            prompt=full_prompt,
            stage_name=refs[0][0],
            filename=refs[0][1],
            temperature=temperature,
        )

    # ------------------------------------------------------------------ #
    #  Stage upload utilities
    # ------------------------------------------------------------------ #

    def _ensure_stage(self, stage_name: str):
        """Create an internal stage if it does not already exist.

        Args:
            stage_name: Snowflake stage identifier.
        """
        if stage_name in self._stage_created:
            return
        self._connect()
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                f"CREATE STAGE IF NOT EXISTS {stage_name} "
                f"ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')"
            )
            self._stage_created.add(stage_name)
        finally:
            cursor.close()

    def upload_to_stage(
        self,
        local_path: str,
        stage_name: str = "cortex_media_stage",
    ) -> str:
        """Upload a local file to a Snowflake internal stage.

        Args:
            local_path: Absolute path to the local file.
            stage_name: Target stage identifier.

        Returns:
            Stage reference string (e.g. ``@cortex_media_stage/image.png``).

        Raises:
            ValueError: If *stage_name* contains invalid characters.
        """
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_.]*$", stage_name):
            raise ValueError(f"Invalid stage name: {stage_name}")

        self._ensure_stage(stage_name)

        self._connect()
        cursor = self.connection.cursor()
        try:
            normalized = local_path.replace("\\", "/")
            cursor.execute(
                f"PUT 'file://{normalized}' @{stage_name} "
                f"AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
            )
            filename = os.path.basename(local_path)
            return f"@{stage_name}/{filename}"
        finally:
            cursor.close()

    def upload_bytes_to_stage(
        self,
        data: bytes,
        filename: str,
        stage_name: str = "cortex_media_stage",
    ) -> str:
        """Upload raw bytes to a Snowflake stage via a temporary file.

        Args:
            data: Raw file content.
            filename: Desired filename (used for extension detection).
            stage_name: Target stage identifier.

        Returns:
            Stage reference string.
        """
        ext = os.path.splitext(filename)[1] or ".bin"
        with tempfile.NamedTemporaryFile(
            suffix=ext,
            prefix="sf_upload_",
            delete=False,
        ) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            url = self.upload_to_stage(tmp_path, stage_name)
            return url
        finally:
            os.unlink(tmp_path)

    def call_with_image(
        self,
        model: str,
        prompt: str,
        stage_name: str,
        filename: str,
        temperature: float = 0.0,
    ) -> str:
        """Invoke COMPLETE with ``TO_FILE()`` for a stage-hosted image.

        Args:
            model: Model identifier.
            prompt: User prompt text.
            stage_name: Snowflake stage containing the image.
            filename: Image filename within the stage.
            temperature: Sampling temperature.

        Returns:
            Generated response string.
        """
        cursor = None
        try:
            self._connect()
            cursor = self.connection.cursor()
            sql = (
                "SELECT SNOWFLAKE.CORTEX.COMPLETE("
                "%s, %s, TO_FILE(%s, %s)"
                ") AS response"
            )
            cursor.execute(sql, (model, prompt, f"@{stage_name}", filename))
            result = cursor.fetchone()
            if result:
                raw = result[0]
                try:
                    obj = json.loads(raw) if isinstance(raw, str) else raw
                    if isinstance(obj, dict) and "choices" in obj:
                        return obj["choices"][0].get("messages", raw)
                except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                    pass
                return raw
            return "No response received from Cortex."
        except snowflake.connector.errors.DatabaseError as e:
            return f"Database Error: {str(e)}"
        except Exception as e:
            return f"An error occurred: {str(e)}"
        finally:
            if cursor is not None:
                cursor.close()


if __name__ == "__main__":

    cortex_client = SnowflakeCortexClient()

    response = cortex_client.call(
        model=os.getenv("SWARM_MODEL", "openai-gpt-5.2"),
        system_prompt="You are a teacher explaining hard concepts in simple terms.",
        user_prompt="Who is GPT?",
        temperature=0.0,
    )
    print("[text-only]", response)

    cortex_client.close()
