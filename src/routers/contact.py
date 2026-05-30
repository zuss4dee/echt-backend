"""B2B contact / clearance form — Notion CRM lead sync."""

from __future__ import annotations

from fastapi import APIRouter, status
from pydantic import BaseModel

from services.notion_crm import notion_crm

router = APIRouter(prefix="/api/contact", tags=["contact"])


class ContactInquiry(BaseModel):
    corporate_entity: str
    portfolio_volume: str
    operational_region: str
    preferred_contact: str
    inquiry_details: str


@router.post("/submit", status_code=status.HTTP_201_CREATED)
async def submit_contact_inquiry(inquiry: ContactInquiry) -> dict[str, str]:
    await notion_crm.create_lead(inquiry)
    return {
        "status": "success",
        "message": "Inquiry successfully received and initialized.",
    }
