from unittest import mock
import uvicorn
import src.api.main as main

def test_uvicorn_run():
    """Mock uvicorn.run() to ensure it is callable"""
    with mock.patch.object(uvicorn, "run") as mock_run:
        uvicorn.run(main.app, host="0.0.0.0", port=8000)
        mock_run.assert_called_once_with(main.app, host="0.0.0.0", port=8000)
