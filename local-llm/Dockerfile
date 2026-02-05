FROM rocm/dev-ubuntu-24.04:7.2-complete

# Install build dependencies
RUN apt-get update && apt-get install -y \
    git cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

# Clone and build llama.cpp with ROCm/HIP support
WORKDIR /build
RUN git clone https://github.com/ggerganov/llama.cpp.git && \
    cd llama.cpp && \
    cmake -B build \
        -DGGML_HIP=ON \
        -DAMDGPU_TARGETS=gfx1030 \
        -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --config Release -j$(nproc)

# Copy binaries to final location
RUN cp /build/llama.cpp/build/bin/* /usr/local/bin/

WORKDIR /models
EXPOSE 8080

# Default: run server
ENTRYPOINT ["llama-server"]
CMD ["--host", "0.0.0.0", "--port", "8080"]
