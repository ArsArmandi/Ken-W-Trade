FROM python:3.12-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    bash \
    procps \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*



# Create a non-root user
RUN groupadd -r vscode && useradd -r -g vscode -d /home/vscode -s /bin/bash vscode \
    && mkdir -p /home/vscode && chown vscode:vscode /home/vscode

# Create opencode config directory and config file for Ollama
RUN mkdir -p /home/vscode/.config/opencode && cat > /home/vscode/.config/opencode/opencode.json << 'EOF'
{
  "$schema": "https://opencode.ai/config.json",
  "model": "Qwen2.5.1-Coder-7B-Instruct-Q4_K_M.gguf",
  "provider": {
    "llamacpp": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "llamacpp (local)",
      "options": {
        "baseURL": "http://host.docker.internal:8080/v1"
      },
      "models": {
        "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf": {
          "name": "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"
        },
        "Qwen2.5.1-Coder-7B-Instruct-Q4_K_M.gguf": {
          "name": "Qwen2.5.1-Coder-7B-Instruct-Q4_K_M.gguf"
        },
        "Qwen2.5.1-Coder-7B-Instruct-Q5_K_M.gguf": {
          "name": "Qwen2.5.1-Coder-7B-Instruct-Q5_K_M.gguf"
        },
        "Qwen2.5-Coder-32B-Instruct-Q3_K_S.gguf": {
          "name": "Qwen2.5-Coder-32B-Instruct-Q3_K_S.gguf"
        },
        "glm-4.7-flash-claude-4.5-opus.q3_k_m": {
          "name": "glm-4.7-flash-claude-4.5-opus.q3_k_m"
        },
        "GLM-4.7-Flash-Q3_K_M.gguf": {
          "name": "GLM-4.7-Flash-Q3_K_M.gguf"
        },
        "GLM-4.7-Flash-Uncen-Hrt-NEO-CODE-MAX-imat-D_AU-IQ3_M": {
          "name": "GLM-4.7-Flash-Uncen-Hrt-NEO-CODE-MAX-imat-D_AU-IQ3_M.gguf"
        },
        "Mistral-Nemo-2407-12B-Thinking-Claude-Gemini-GPT5.2-Uncensored-HERETIC.Q8_0": {
          "name": "Mistral-Nemo-2407-12B-Thinking-Claude-Gemini-GPT5.2-Uncensored-HERETIC.Q8_0.gguf"
        },
        "Qwen3-14B-Claude-4.5-Opus-Distill.q4_k_m": {
          "name": "Qwen3-14B-Claude-4.5-Opus-Distill.q4_k_m.gguf"
        },
        "ruvltra-claude-code-0.5b-q4_k_m": {
          "name": "ruvltra-claude-code-0.5b-q4_k_m.gguf"
        },
        "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M": {
          "name": "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf"
        }
      }
    }
  }
}
EOF

# Set ownership of the config file
RUN chown -R vscode:vscode /home/vscode/.config
RUN mkdir -p /home/vscode/.local/share/opencode
RUN chown -R vscode:vscode /home/vscode/.local/share/opencode

USER vscode

# Install opencode CLI
RUN curl -fsSL https://opencode.ai/install | bash

ENV PATH="/home/vscode/.opencode/bin:${PATH}"

RUN npx -y gsd-opencode@latest --global

# Copy and install Python project dependencies if present
WORKDIR /workspace
COPY . .
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi
RUN if [ -f pyproject.toml ]; then pip install --no-cache-dir -e .; fi




