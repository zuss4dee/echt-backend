"""
Post-payment agency registration: Polar checkout verification + Supabase provisioning.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional

import httpx
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, EmailStr, Field
from supabase import Client, create_client

logger = logging.getLogger(__name__)

_supabase_client: Optional[Client] = None

router = APIRouter(prefix="/api/auth", tags=["auth"])

MIN_PASSWORD_LENGTH = 12
POLAR_API_BASE = "https://api.polar.sh/v1"
SETUP_REDIRECT = "/setup"

PAID_CHECKOUT_STATUSES = frozenset(
    {"succeeded", "confirmed", "paid", "complete", "completed"}
)


class RegisterAgencyRequest(BaseModel):
    agency_name: str = Field(min_length=1, max_length=200)
    director_name: str = Field(min_length=1, max_length=200)
    email: EmailStr
    password: str = Field(min_length=MIN_PASSWORD_LENGTH)
    session_id: str = Field(min_length=1, max_length=256)


class RegisterAgencyResponse(BaseModel):
    ok: bool = True
    status: str = "success"
    message: str = "Workspace successfully initialized."
    agency: str
    user_id: str
    redirect: str = SETUP_REDIRECT


class CheckoutEmailResponse(BaseModel):
    email: Optional[str] = None


def _get_supabase() -> Client:
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    url = re.sub(r"\s+", "", os.getenv("SUPABASE_URL") or "")
    key = re.sub(r"\s+", "", os.getenv("SUPABASE_KEY") or "")
    if not url or not key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database is not configured.",
        )

    _supabase_client = create_client(url, key)
    return _supabase_client


def _env_bool(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in ("1", "true", "yes")


def _polar_token() -> Optional[str]:
    raw = os.getenv("POLAR_SECRET_KEY") or os.getenv("POLAR_ACCESS_TOKEN") or ""
    token = re.sub(r"\s+", "", raw)
    return token or None


def _checkout_is_paid(checkout: dict[str, Any]) -> bool:
    status_value = str(checkout.get("status") or "").lower()
    return status_value in PAID_CHECKOUT_STATUSES


def _checkout_email(checkout: dict[str, Any]) -> Optional[str]:
    for key in ("customer_email", "customerEmail", "email"):
        value = checkout.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _checkout_customer_id(checkout: dict[str, Any]) -> Optional[str]:
    for key in ("customer_id", "customerId"):
        value = checkout.get(key)
        if value is not None:
            return str(value)
    return None


async def fetch_polar_checkout(session_id: str) -> dict[str, Any]:
    """Resolve a Polar checkout by ID (hosted link or API-created session)."""
    token = _polar_token()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Polar is not configured on the server.",
        )

    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    urls = (
        f"{POLAR_API_BASE}/checkouts/{session_id}",
        f"{POLAR_API_BASE}/checkouts/custom/{session_id}",
    )

    last_status = 0
    async with httpx.AsyncClient(timeout=20.0) as client:
        for url in urls:
            response = await client.get(url, headers=headers)
            last_status = response.status_code
            if response.status_code == 200:
                return response.json()
            if response.status_code == 404:
                continue

    logger.warning(
        "Polar checkout lookup failed for session_id=%s last_status=%s",
        session_id,
        last_status,
    )
    raise HTTPException(
        status_code=status.HTTP_402_PAYMENT_REQUIRED,
        detail="Invalid or unpaid checkout session.",
    )


async def verify_paid_checkout(session_id: str) -> dict[str, Any]:
    if _env_bool("POLAR_SKIP_VERIFY"):
        logger.warning("POLAR_SKIP_VERIFY is enabled — skipping checkout validation.")
        return {}

    checkout = await fetch_polar_checkout(session_id)
    if not _checkout_is_paid(checkout):
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Checkout session has not been completed successfully.",
        )
    return checkout


def provision_supabase_user(
    supabase: Client,
    *,
    email: str,
    password: str,
    agency_name: str,
    director_name: str,
    session_id: str,
    polar_customer_id: Optional[str],
) -> str:
    metadata: dict[str, Any] = {
        "full_name": director_name,
        "company_name": agency_name,
        "polar_checkout_id": session_id,
        "onboarding_complete": False,
        "subscription_status": "active",
    }
    if polar_customer_id:
        metadata["polar_customer_id"] = polar_customer_id

    try:
        result = supabase.auth.admin.create_user(
            {
                "email": email,
                "password": password,
                "email_confirm": True,
                "user_metadata": metadata,
            }
        )
    except Exception as exc:
        message = str(exc).lower()
        if "already" in message or "registered" in message or "exists" in message:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An account with this email already exists. Sign in instead.",
            ) from exc
        logger.exception("Supabase create_user failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not create account.",
        ) from exc

    user = getattr(result, "user", None)
    user_id = getattr(user, "id", None) if user is not None else None
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not create account.",
        )

    profile_row: dict[str, Any] = {
        "id": user_id,
        "email": email,
        "full_name": director_name,
        "company_name": agency_name,
        "onboarding_complete": False,
    }
    optional_profile_fields = {
        "polar_checkout_id": session_id,
        "polar_customer_id": polar_customer_id,
        "subscription_status": "active",
    }
    for key, value in optional_profile_fields.items():
        if value is not None:
            profile_row[key] = value

    try:
        supabase.table("profiles").upsert(profile_row, on_conflict="id").execute()
    except Exception as exc:
        # Auth user exists; profile columns may be missing until migration is applied.
        logger.warning("profiles upsert failed (non-fatal): %s", exc)

    return str(user_id)


@router.get("/register", response_model=CheckoutEmailResponse)
async def register_checkout_email(
    session_id: str = Query(..., min_length=1),
) -> CheckoutEmailResponse:
    """Optional email prefill from a completed Polar checkout."""
    if _env_bool("POLAR_SKIP_VERIFY"):
        return CheckoutEmailResponse(email=None)

    try:
        checkout = await fetch_polar_checkout(session_id)
    except HTTPException:
        return CheckoutEmailResponse(email=None)

    return CheckoutEmailResponse(email=_checkout_email(checkout))


@router.post(
    "/register",
    status_code=status.HTTP_201_CREATED,
    response_model=RegisterAgencyResponse,
)
async def register_agency(request: RegisterAgencyRequest) -> RegisterAgencyResponse:
    """Verify Polar payment, create Supabase auth user, and upsert public.profiles."""
    db = _get_supabase()

    checkout = await verify_paid_checkout(request.session_id.strip())
    polar_customer_id = _checkout_customer_id(checkout)

    email = str(request.email).strip().lower()
    user_id = provision_supabase_user(
        db,
        email=email,
        password=request.password,
        agency_name=request.agency_name.strip(),
        director_name=request.director_name.strip(),
        session_id=request.session_id.strip(),
        polar_customer_id=polar_customer_id,
    )

    return RegisterAgencyResponse(
        agency=request.agency_name.strip(),
        user_id=user_id,
    )
