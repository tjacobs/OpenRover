# OpenRover
# Web.py

# Serves the webpage from which you can control your rover.

# The basics
import os, sys, time, math

# ------ Settings -------


# -----------------------

# Get IP address
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(("8.8.8.8", 80))
print(s.getsockname()[0])
myIP = s.getsockname()[0]
s.close()

# Start
print("Starting Webserver on " + str(myIP) + ".")

# Import
from http.server import BaseHTTPRequestHandler, HTTPServer
import http.server

# HTTPRequestHandler class
class testHTTPServer_RequestHandler(BaseHTTPRequestHandler):
 
  # GET
  def do_GET(self):
        # Send response status code
        self.send_response(200)
 
        # Send headers
        self.send_header('Content-type','text/html')
        self.end_headers()
 
        # Send message back to client
        message = "Hello world!"
        # Write content as utf-8 data
        self.wfile.write(bytes(message, "utf8"))
        return
 
def run():
 
  server_address = (myIP, 80)

  # Serve from web dir
  web_dir = os.path.join(os.path.dirname(__file__), 'web')
  os.chdir(web_dir)

  Handler = http.server.SimpleHTTPRequestHandler
  httpd = HTTPServer(server_address, Handler)
  httpd.serve_forever()
 
run()
