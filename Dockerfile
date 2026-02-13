# ---- Stage 1: Build ----
# We use a multi-stage build to keep the final image small.
# This stage installs dependencies; the final stage copies only what's needed.

FROM python:3.12-slim AS builder

# Set working directory inside the container
WORKDIR /app

# Copy ONLY requirements first — Docker caches this layer.
# If requirements.txt hasn't changed, Docker reuses the cached layer
# instead of re-downloading all packages. This makes rebuilds fast.
COPY requirements.txt .

# Install Python dependencies into a virtual environment
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt


# ---- Stage 2: Runtime ----
# Start fresh from a clean slim image — the builder stage is discarded.
# This means build tools, pip cache, etc. don't bloat the final image.

FROM python:3.12-slim

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Make sure the venv's Python/pip are used (not system ones)
ENV PATH="/opt/venv/bin:$PATH"

# Copy the application code
COPY app/ ./app/

# Don't run as root in production — create a non-root user.
# This limits damage if the container is compromised.
RUN useradd --create-home appuser
USER appuser

# Expose port 8000 (documentation — doesn't actually open the port)
EXPOSE 8000

# Health check — Docker/orchestrators use this to know if the container is healthy
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Start the FastAPI server with uvicorn
# --host 0.0.0.0: Listen on all interfaces (required inside Docker)
# --workers 2: Run 2 worker processes for handling concurrent requests
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
