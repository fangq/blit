FROM quay.io/pypa/manylinux2014_x86_64
RUN yum install -y epel-release && \
    yum install -y suitesparse-devel lapack-devel blas-devel openblas-devel && \
    yum clean all
ENV PATH /opt/rh/devtoolset-7/root/usr/bin:${PATH}
ENV UMFPACK_INCLUDE=/usr/include/suitesparse
ENV UMFPACK_LIB=/usr/lib64
COPY build_wheels.sh /
WORKDIR /src
ENTRYPOINT ["/build_wheels.sh"]