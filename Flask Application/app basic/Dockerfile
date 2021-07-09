FROM python:3.8

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

ENTRYPOINT [ "python3" ]

# install libraries we really need for this
RUN pip3 install -r requirements.txt

#Expose indicates the ports (in this case 8080) that the container should listen to for connections.
EXPOSE 8080

# Run your application!
CMD ["run.py"]
