FROM ubuntu:20.04

# Install General Requirements
RUN apt-get update && \
        apt-get install -y --no-install-recommends \
        python3.9 \
        python3-pip \
        python3-pandas \
        software-properties-common

RUN pip install -q numpy
RUN pip install -q opencv-python-headless

# Create a /work directory within the container, copy everything from the build directory and switch there.
RUN mkdir /work
COPY . /work
WORKDIR /work

# Make sure your scripts are made executable, forgetting this is a common error. 
# Another frequent error is having Windows style line endings in .sh files. 
RUN chmod +x test.sh
RUN chmod +x train.sh

# No need to add any CMD or ENTRYPOINT
