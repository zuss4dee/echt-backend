"""B2B contact / clearance form — Notion CRM lead sync + Slack alerts."""

from __future__ import annotations

import os

import httpx
from fastapi import APIRouter, status
from pydantic import BaseModel

from services.notion_crm import notion_crm

router = APIRouter(prefix="/api/contact", tags=["contact"])
waitlist_router = APIRouter(prefix="/api/waitlist", tags=["waitlist"])


class ContactInquiry(BaseModel):
    corporate_entity: str
    portfolio_volume: str
    operational_region: str
    preferred_contact: str
    inquiry_details: str


def _slack_inquiry_text(inquiry: ContactInquiry) -> str:
    return (
        "*New B2B contact inquiry*\n"
        f"*Corporate entity:* {inquiry.corporate_entity}\n"
        f"*Portfolio volume:* {inquiry.portfolio_volume}\n"
        f"*Operational region:* {inquiry.operational_region}\n"
        f"*Preferred contact:* {inquiry.preferred_contact}\n"
        f"*Inquiry details:*\n{inquiry.inquiry_details}"
    )


@router.post("/submit", status_code=status.HTTP_201_CREATED)
async def submit_contact_inquiry(inquiry: ContactInquiry) -> dict[str, str]:
    await notion_crm.create_lead(inquiry)

    slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if slack_webhook_url:
        payload = {"text": _slack_inquiry_text(inquiry)}
        try:
            async with httpx.AsyncClient() as client:
                await client.post(slack_webhook_url, json=payload)
        except Exception as e:
            print(f"Slack webhook failed: {e}")
    else:
        print("Warning: SLACK_WEBHOOK_URL is not set. Skipping Slack notification.")

    return {
        "status": "success",
        "message": "Inquiry successfully received and initialized.",
    }


class WaitlistEntry(BaseModel):
    email: str


@waitlist_router.post("/submit", status_code=status.HTTP_201_CREATED)
async def submit_waitlist_entry(entry: WaitlistEntry) -> dict[str, str]:
    await notion_crm.add_to_waitlist(entry.email)
    return {
        "status": "success",
        "message": "Waitlist entry successfully received.",
    }
