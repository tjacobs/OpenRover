CXXFLAGS = -std=gnu++0x -Wall
LDLIBS = -lstdc++

.PHONY: all clean test
all: websockets
testserver: node_modules
	node example-server.js
node_modules:
	npm install ws
websockets: websockets.o easywsclient.o
websockets.o: websockets.cpp easywsclient.hpp
easywsclient.o: easywsclient.cpp easywsclient.hpp

#  Makefile for server
webserver:	webserver.cpp
		gcc webserver.cpp -o webserver

