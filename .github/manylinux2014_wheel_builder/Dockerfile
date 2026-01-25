FROM quay.io/pypa/manylinux2014_x86_64

# Install build dependencies
RUN yum install -y epel-release && \
    yum install -y \
        suitesparse-devel \
        lapack-devel \
        blas-devel \
        openblas-devel \
        pkgconfig \
    && yum clean all

# Set environment variables for library discovery
ENV PATH=/opt/rh/devtoolset-7/root/usr/bin:${PATH}
ENV UMFPACK_INCLUDE=/usr/include/suitesparse
ENV UMFPACK_LIB=/usr/lib64
ENV PKG_CONFIG_PATH=/usr/lib64/pkgconfig

# Copy build script
COPY build_wheels.sh /
RUN chmod +x /build_wheels.sh

WORKDIR /src
ENTRYPOINT ["/build_wheels.sh"]