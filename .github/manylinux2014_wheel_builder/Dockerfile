FROM quay.io/pypa/manylinux2014_x86_64

# Install build dependencies
RUN yum install -y epel-release && \
    yum install -y \
        suitesparse-devel \
        lapack-devel \
        blas-devel \
        openblas-devel \
        gcc-gfortran \
        pkgconfig \
    && yum clean all

# Create OpenBLAS pkg-config file if missing (common issue on manylinux2014)
RUN if ! pkg-config --exists openblas 2>/dev/null; then \
        mkdir -p /usr/lib64/pkgconfig && \
        echo 'prefix=/usr' > /usr/lib64/pkgconfig/openblas.pc && \
        echo 'libdir=${prefix}/lib64' >> /usr/lib64/pkgconfig/openblas.pc && \
        echo 'includedir=${prefix}/include/openblas' >> /usr/lib64/pkgconfig/openblas.pc && \
        echo '' >> /usr/lib64/pkgconfig/openblas.pc && \
        echo 'Name: OpenBLAS' >> /usr/lib64/pkgconfig/openblas.pc && \
        echo 'Description: OpenBLAS library' >> /usr/lib64/pkgconfig/openblas.pc && \
        echo 'Version: 0.3.0' >> /usr/lib64/pkgconfig/openblas.pc && \
        echo 'Libs: -L${libdir} -lopenblas' >> /usr/lib64/pkgconfig/openblas.pc && \
        echo 'Cflags: -I${includedir}' >> /usr/lib64/pkgconfig/openblas.pc; \
    fi

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