import os


BASE_CONFIG = {
    "uploaded_pdf_path": "data/uploaded_papers",
    "processed_paper_path": "data/processed_papers",
    "llm_api_keys": {},
    "mathpix_credentials": {
        "app_id": os.environ.get("MATHPIX_APP_ID", ""),
        "app_key": os.environ.get("MATHPIX_APP_KEY", ""),
    },
    "claude_api_key": os.environ.get("ANTHROPIC_API_KEY","")
}


docker_config = {
    **BASE_CONFIG,
    "grobid_url": "http://collage-grobid-1:8070",
    "chemdataextractor_service_url": "http://collage-chemdataextractor-1:8000",
}

sireesh_dev_config = {
    **BASE_CONFIG,
    "grobid_url": "http://windhoek.sp.cs.cmu.edu:8070",
    "chemdataextractor_service_url": "http://windhoek.sp.cs.cmu.edu:8001",
}

configs = {"docker": docker_config, "sireesh_dev": sireesh_dev_config}

default_config = docker_config

app_config = configs.get(os.environ.get("CONFIG_NAME"), default_config)
