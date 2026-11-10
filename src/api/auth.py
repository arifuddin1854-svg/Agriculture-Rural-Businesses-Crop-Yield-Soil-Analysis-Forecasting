"""
OAuth2 & JWT Authentication Module for Axlero Agri-Tech Platform.
Deliverable: src/api/auth.py (Engineer 3)

Provides:
- Cryptographic password hashing and verification using PBKDF2-HMAC-SHA256
- OAuth2 Password Bearer flow for secure API and dashboard access
- JWT access token generation, validation, and claims extraction
- FastAPI authentication dependencies for route protection and user identity injection
- Authentication router for user registration, login, and profile retrieval
"""

from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import logging
import os
import secrets
from typing import Any, Dict, Optional
import uuid

import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from src.api.config import settings
from src.api.database import get_db
from src.api.models.user import User
from src.api.schemas.user import UserCreate, UserRead

logger = logging.getLogger("axlero_agri_api.auth")

# OAuth2 Scheme configured to token endpoint
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/auth/token",
    auto_error=False,
)

# Configuration Constants
SECRET_KEY = getattr(
    settings, "SECRET_KEY", "axlero_agri_super_secure_jwt_token_secret_key_2026_9b8c"
)
ALGORITHM = getattr(settings, "JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = getattr(settings, "ACCESS_TOKEN_EXPIRE_MINUTES", 1440)
PBKDF2_ITERATIONS = 100_000


# ---------------------------------------------------------------------------
# Password Hashing & Verification
# ---------------------------------------------------------------------------


def get_password_hash(password: str) -> str:
    """
    Hashes a plaintext password using salted PBKDF2-HMAC-SHA256 with 100,000 rounds.
    Output format: pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>
    """
    salt = secrets.token_bytes(16)
    key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${key.hex()}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verifies a plaintext password against a stored hashed password string.
    Supports PBKDF2-HMAC-SHA256 standard and handles backward-compatible legacy formats.
    """
    if not hashed_password:
        return False

    parts = hashed_password.split("$")
    if parts[0] == "pbkdf2_sha256":
        # Form 1: pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>
        if len(parts) == 4:
            try:
                iterations = int(parts[1])
                salt = bytes.fromhex(parts[2])
                stored_hash = parts[3]
                key = hashlib.pbkdf2_hmac(
                    "sha256",
                    plain_password.encode("utf-8"),
                    salt,
                    iterations,
                )
                return hmac.compare_digest(key.hex(), stored_hash)
            except Exception as e:
                logger.warning(f"Error validating PBKDF2 hash: {e}")
                return False
        # Form 2: Legacy placeholder format pbkdf2_sha256$<plaintext>
        elif len(parts) == 2:
            return hmac.compare_digest(plain_password, parts[1])

    # Fallback constant-time comparison
    return hmac.compare_digest(plain_password, hashed_password)


# ---------------------------------------------------------------------------
# JWT Token Issuance & Decoding
# ---------------------------------------------------------------------------


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Generates a cryptographically signed JWT bearer token containing user claims.
    """
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update(
        {
            "exp": expire,
            "iat": now,
            "nbf": now,
            "jti": str(uuid.uuid4()),
        }
    )
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# Default refresh token lifetime: 7 days
REFRESH_TOKEN_EXPIRE_DAYS = 7


def create_refresh_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Generates a long-lived JWT refresh token (default 7 days).
    Includes ``type: "refresh"`` claim to distinguish from access tokens.
    Refresh tokens are stateless — no DB storage required.
    """
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    to_encode.update(
        {
            "exp": expire,
            "iat": now,
            "nbf": now,
            "type": "refresh",
            "jti": str(uuid.uuid4()),
        }
    )
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> Dict[str, Any]:
    """
    Decodes and validates a JWT token signature and expiration.
    Raises HTTPException 401 on expired or tampered tokens.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token has expired. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or malformed authentication token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------------------
# Authentication Schemas
# ---------------------------------------------------------------------------


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_seconds: int
    user: UserRead
    refresh_token: Optional[str] = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1)


class RefreshTokenRequest(BaseModel):
    """Request body for token rotation — supply a valid refresh JWT."""

    refresh_token: str = Field(
        ..., description="A valid refresh token issued by /auth/token or /auth/login."
    )


# ---------------------------------------------------------------------------
# User Authentication Helpers & Dependencies
# ---------------------------------------------------------------------------


def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """
    Queries user by email and validates against stored password hash.
    """
    user = db.query(User).filter(User.email == email.lower().strip()).first()
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    FastAPI dependency: Resolves the current authenticated User from the JWT Bearer token.
    Raises HTTP 401 if missing, invalid, or user does not exist.
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(token)
    user_id_str = payload.get("user_id") or payload.get("sub")
    if not user_id_str:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token payload is missing user identity.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        user_uuid = uuid.UUID(str(user_id_str))
        user = db.query(User).filter(User.id == user_uuid).first()
    except (ValueError, TypeError):
        user = db.query(User).filter(User.email == str(user_id_str)).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user no longer exists in system.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    FastAPI dependency: Asserts that authenticated user account is active.
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user account.",
        )
    return current_user


def get_optional_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """
    FastAPI dependency: Optionally extracts authenticated user if token present and valid;
    returns None for anonymous/guest sessions without erroring.
    """
    if not token:
        return None
    try:
        payload = decode_access_token(token)
        user_id_str = payload.get("user_id") or payload.get("sub")
        if not user_id_str:
            return None
        try:
            user_uuid = uuid.UUID(str(user_id_str))
            return db.query(User).filter(User.id == user_uuid).first()
        except (ValueError, TypeError):
            return db.query(User).filter(User.email == str(user_id_str)).first()
    except HTTPException:
        return None


# ---------------------------------------------------------------------------
# Router: /auth (Login, Token, Register, Me)
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/auth", tags=["Authentication & User Sessions"])


@router.post(
    "/token",
    response_model=Token,
    summary="OAuth2 Password Flow Token Issuance",
)
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    """
    OAuth2 standard password token exchange. Ingests form-encoded username & password.
    Returns signed JWT access token and user profile.
    """
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User account is deactivated.",
        )

    expires_seconds = ACCESS_TOKEN_EXPIRE_MINUTES * 60
    _token_data = {
        "sub": str(user.id),
        "user_id": str(user.id),
        "email": user.email,
        "role": user.role,
    }
    access_token = create_access_token(
        data=_token_data,
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    refresh_token = create_refresh_token(data=_token_data)

    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in_seconds=expires_seconds,
        user=UserRead.model_validate(user),
        refresh_token=refresh_token,
    )


@router.post(
    "/login",
    response_model=Token,
    summary="JSON Login for Secure Dashboard Access",
)
def login_json(
    credentials: LoginRequest,
    db: Session = Depends(get_db),
):
    """
    JSON credentials login for web dashboards, mobile apps, and programmatic REST clients.
    """
    user = authenticate_user(db, credentials.email, credentials.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User account is deactivated.",
        )

    expires_seconds = ACCESS_TOKEN_EXPIRE_MINUTES * 60
    _token_data = {
        "sub": str(user.id),
        "user_id": str(user.id),
        "email": user.email,
        "role": user.role,
    }
    access_token = create_access_token(
        data=_token_data,
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    refresh_token = create_refresh_token(data=_token_data)

    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in_seconds=expires_seconds,
        user=UserRead.model_validate(user),
        refresh_token=refresh_token,
    )


@router.post(
    "/register",
    response_model=Token,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user and receive JWT session",
)
def register_user(
    user_in: UserCreate,
    db: Session = Depends(get_db),
):
    """
    Registers a new farmer, agronomist, or business manager account with salted password hashing,
    and immediately issues a valid JWT access token.
    """
    clean_email = user_in.email.lower().strip()
    existing = db.query(User).filter(User.email == clean_email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email address already exists.",
        )

    hashed_pw = get_password_hash(user_in.password)
    user_data = user_in.model_dump(exclude={"password"})
    user_data["email"] = clean_email
    db_user = User(**user_data, hashed_password=hashed_pw)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    expires_seconds = ACCESS_TOKEN_EXPIRE_MINUTES * 60
    _token_data = {
        "sub": str(db_user.id),
        "user_id": str(db_user.id),
        "email": db_user.email,
        "role": db_user.role,
    }
    access_token = create_access_token(
        data=_token_data,
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    refresh_token = create_refresh_token(data=_token_data)

    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in_seconds=expires_seconds,
        user=UserRead.model_validate(db_user),
        refresh_token=refresh_token,
    )


@router.get(
    "/me",
    response_model=UserRead,
    summary="Get Current Authenticated User Profile",
)
def get_me(current_user: User = Depends(get_current_active_user)):
    """
    Returns the profile and farm details of the currently authenticated JWT session.
    """
    return current_user


@router.post(
    "/refresh",
    response_model=Token,
    summary="JWT Token Rotation — Exchange Refresh Token for New Access Token",
    responses={
        200: {"description": "New access token and rotated refresh token issued."},
        401: {
            "description": (
                "Refresh token is invalid, expired, or not a refresh-type token."
            )
        },
    },
)
def refresh_access_token(
    body: RefreshTokenRequest,
    db: Session = Depends(get_db),
):
    """
    Token rotation endpoint for secure session persistence on mobile and dashboard clients.

    Supply a valid ``refresh_token`` (issued by ``/auth/token`` or ``/auth/login``).
    Returns a fresh **access token** (short-lived) and a new **refresh token** (rotation).

    Rejects:
    - Expired refresh tokens (HTTP 401)
    - Access tokens submitted as refresh tokens (HTTP 401)
    - Tokens for non-existent users (HTTP 401)
    """
    _unauth = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired refresh token.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(body.refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has expired. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError:
        raise _unauth

    # Enforce token-type guard: reject access tokens submitted here
    if payload.get("type") != "refresh":
        raise _unauth

    user_id_str = payload.get("user_id") or payload.get("sub")
    if not user_id_str:
        raise _unauth

    try:
        user_uuid = uuid.UUID(str(user_id_str))
        user = db.query(User).filter(User.id == user_uuid).first()
    except (ValueError, TypeError):
        user = db.query(User).filter(User.email == str(user_id_str)).first()

    if not user or not user.is_active:
        raise _unauth

    expires_seconds = ACCESS_TOKEN_EXPIRE_MINUTES * 60
    _token_data = {
        "sub": str(user.id),
        "user_id": str(user.id),
        "email": user.email,
        "role": user.role,
    }
    new_access_token = create_access_token(
        data=_token_data,
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    new_refresh_token = create_refresh_token(data=_token_data)

    logger.info("Token rotated for user %s (id=%s)", user.email, user.id)

    return Token(
        access_token=new_access_token,
        token_type="bearer",
        expires_in_seconds=expires_seconds,
        user=UserRead.model_validate(user),
        refresh_token=new_refresh_token,
    )
