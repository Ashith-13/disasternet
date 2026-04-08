import os
from fastapi.responses import HTMLResponse
from openenv.core.env_server import create_fastapi_app
from models import DisasterAction, DisasterObservation

try:
    from .environment import DisasterNetEnvironment
except ImportError:
    from environment import DisasterNetEnvironment

os.environ.setdefault("ENABLE_WEB_INTERFACE", "true")

app = create_fastapi_app(
    DisasterNetEnvironment,
    DisasterAction,
    DisasterObservation
)

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>DisasterNET</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: #0b1220;
                color: white;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                text-align: center;
            }
            .card {
                background: #111827;
                padding: 40px;
                border-radius: 16px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                max-width: 700px;
            }
            h1 { color: #22c55e; margin-bottom: 12px; }
            p { color: #d1d5db; line-height: 1.6; }
            a {
                display: inline-block;
                margin-top: 18px;
                padding: 12px 20px;
                background: #22c55e;
                color: black;
                text-decoration: none;
                border-radius: 10px;
                font-weight: bold;
            }
            a:hover { background: #16a34a; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🌍 DisasterNET</h1>
            <p><strong>Multi-Agency Disaster Response Coordination Environment</strong></p>
            <p>
                DisasterNET is an OpenEnv-compatible environment for training AI agents
                to coordinate rescue, medical, engineering, and communication resources
                during the critical 72-hour disaster response window.
            </p>
            <p>
                Use the OpenAPI docs to test the environment endpoints directly.
            </p>
            <a href="/docs">Open API Docs</a>
        </div>
    </body>
    </html>
    """

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()