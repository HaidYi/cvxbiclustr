FROM ubuntu:18.04

# create workdir
RUN mkdir -p /app
WORKDIR /app

# install basic utilities
RUN apt-get update && apt-get install -y \
  git \
  wget \
  build-essential \
  libxml2-dev \
  libgmp-dev \
  libopenblas-dev \
  autoconf \
  automake \
  libtool \
  python3-pip

# download igraph
RUN wget https://github.com/igraph/igraph/releases/download/0.8.2/igraph-0.8.2.tar.gz \
  && tar -xzvf igraph-0.8.2.tar.gz

# compile and install igraph
RUN cd igraph-0.8.2 \
  && ./configure \
  && make -j 4\
  && make check \
  && make install

# rm unnecessary files
RUN rm -rf /app/igraph-0.8.2 /app/igraph-0.8.2.tar.gz

# download and install cvxbiclustr library
RUN git clone https://github.com/haidyi/cvxbiclustr.git \
 && cd cvxbiclustr/lib/cvxclustr-0.3 \
 && autoreconf -i \
 && ./configure \
 && make \
 && make install \
 && make clean

# install required python packages
WORKDIR /app/cvxbiclustr
RUN pip3 install -r requirements.txt

# generate the cvxclustr_path bin file
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
RUN make

# CMD
CMD ["python3", "cvxclustr.py", "--gamma", "3,5,10,20,50,100,150,200,500,1000", "--col_knn", "2", "--row_knn", "4", "--tol", "0.001", "--data", "data/president.csv", "--output", "data/president.json"]
