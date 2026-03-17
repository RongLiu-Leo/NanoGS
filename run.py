# run_server.py
import http.server
import socketserver

class MyHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Force the correct header for module scripts
        if self.path.endswith('.js') or self.path.endswith('.mjs'):
            self.send_header('Content-Type', 'text/javascript')
        super().end_headers()

# Add mappings for common extensions
MyHandler.extensions_map.update({
    '.js': 'text/javascript',
    '.mjs': 'text/javascript',
})

PORT = 8010
with socketserver.TCPServer(("", PORT), MyHandler) as httpd:
    print(f"Serving at http://localhost:{PORT}")
    httpd.serve_forever()
