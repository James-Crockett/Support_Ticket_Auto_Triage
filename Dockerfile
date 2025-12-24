# 1. Base image
FROM python:3.9-slim

# 2. Install uv
RUN pip install uv

# 3. Set working directory
WORKDIR /app

# 4. Copy dependency files first (for caching)
COPY pyproject.toml uv.lock ./

# 5. Install dependencies
RUN uv sync --frozen
ENV PATH="/app/.venv/bin:$PATH"

# 6. COPY EVERYTHING
# Instead of picking files one by one, we copy the whole project.
# Your .dockerignore file will automatically exclude .venv, data, notebooks, etc.
COPY . .

# 7. Expose the port
EXPOSE 8000

# 8. Run the application
# Since main.py is in the root, we call "main:app"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]