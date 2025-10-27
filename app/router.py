from fastapi import APIRouter

import endpoint


router = APIRouter()

router.include_router(endpoint.router, prefix="/chat", tags=["chat"])
