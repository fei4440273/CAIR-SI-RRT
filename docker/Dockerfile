FROM ubuntu:24.04@sha256:019e8eb29a85e74d64925745884f2ec79aa27e3feab36353d24656f4d6b89467

COPY install.sh /usr/local/src/cair-si-rrt/install.sh

RUN USER=root \
    SIRRT_INSTALL_PHASE=system \
    bash /usr/local/src/cair-si-rrt/install.sh \
    && rm -rf /var/lib/apt/lists/*

ARG CAIR_BUILD_JOBS

COPY docker/git-retry.sh /usr/local/lib/cair-si-rrt/git
COPY docker/install-dependencies.sh /usr/local/lib/cair-si-rrt/install-dependencies
RUN chmod 0755 /usr/local/lib/cair-si-rrt/git /usr/local/lib/cair-si-rrt/install-dependencies

RUN --mount=type=cache,target=/var/cache/sirrt-archives \
    USER=root \
    PATH="/usr/local/lib/cair-si-rrt:${PATH}" \
    SIRRT_INSTALL_PHASE=dependencies \
    SIRRT_JOBS="${CAIR_BUILD_JOBS}" \
    RELEASE_C_FLAGS="-O3 -DNDEBUG -fno-plt" \
    RELEASE_CXX_FLAGS="-O3 -DNDEBUG -fno-plt" \
    DEBUG_C_FLAGS="-O0 -g -fno-omit-frame-pointer" \
    DEBUG_CXX_FLAGS="-O0 -g -fno-omit-frame-pointer" \
    bash /usr/local/lib/cair-si-rrt/install-dependencies \
    && rm -rf /opt/SIRRT_dependencies/src

WORKDIR /workspace
CMD ["bash"]

