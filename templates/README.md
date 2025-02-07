# Home Automation Control Panel

This Flask application serves as a control panel for home automation, featuring device status monitoring and IoT control capabilities. The application is designed to work with Cloudflare Tunnels for secure remote access.

## Architecture Overview

### Frontend Structure
The application uses a monolithic HTML structure where CSS and JavaScript are included directly in the HTML templates rather than as separate files. This design choice is intentional and necessary due to the use of Cloudflare Tunnels for remote access.

### Why Include Scripts and Styles in HTML?

When using Cloudflare Tunnels, all resources need to be served in a way that ensures they're accessible through the tunnel. By including CSS and JavaScript directly in the HTML templates:

1. **Reduced Latency**: All necessary code is delivered in a single request, eliminating the need for multiple HTTP requests.
2. **Tunnel Compatibility**: Ensures all resources are properly transmitted through the Cloudflare Tunnel without path resolution issues.
3. **Simplified Deployment**: No need to configure additional static file serving rules in the Cloudflare Tunnel configuration.
4. **Reliability**: Prevents potential issues with resource loading that could occur when separate files are requested through the tunnel.

### File Structure
```
.
├── templates/
│   ├── index.html        # Main template with embedded CSS/JS
│   ├── login.html        # Login page with embedded styles
│   └── button_control.html # Control interface
├── app.py               # Main Flask application
└── .env                # Environment variables (not in repo)
```

## Setup and Configuration

### Environment Variables
Create a `.env` file with the following variables:
```
MERAKI_API_TOKEN=your_token_here
SECRET_KEY=your_secret_key_here
ROUTINE_API_TOKEN=your_routine_token_here
BUTTON_PAGE_PASSWORDS=password1,password2
HOST=your_host_ip
PORT=your_port_number
```

### Cloudflare Tunnel Setup

1. Install cloudflared
2. Authenticate with Cloudflare:
   ```bash
   cloudflared tunnel login
   ```
3. Create a tunnel:
   ```bash
   cloudflared tunnel create <tunnel-name>
   ```
4. Configure your tunnel to point to your Flask application:
   ```yaml
   tunnel: <tunnel-id>
   credentials-file: /path/to/credentials.json
   ingress:
     - hostname: your.domain.com
       service: http://localhost:5000
     - service: http_status:404
   ```
5. Start the tunnel:
   ```bash
   cloudflared tunnel run <tunnel-name>
   ```

## Security Considerations

- All sensitive information is stored in environment variables
- Authentication is required for accessing control features
- Cloudflare Tunnels provide secure remote access without opening ports
- The application uses HTTPS through Cloudflare's SSL/TLS encryption

## Development Notes

When making changes to the frontend:
1. Edit the HTML templates directly
2. CSS styles should be placed in `<style>` tags within the HTML
3. JavaScript should be placed in `<script>` tags within the HTML
4. Avoid creating separate .css or .js files as they may not be served correctly through the tunnel

## Troubleshooting

If you experience issues with the application:

1. Check Cloudflare Tunnel logs:
   ```bash
   cloudflared tunnel run <tunnel-name> --loglevel debug
   ```
2. Verify all environment variables are set correctly
3. Ensure the Flask application is running and accessible locally
4. Check the Flask application logs for any errors

## Contributing

1. Fork the repository
2. Create your feature branch
3. Make your changes
4. Submit a pull request

Remember to never commit sensitive information or .env files to the repository.
