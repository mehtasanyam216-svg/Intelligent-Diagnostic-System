from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


def utcnow():
    return datetime.now(timezone.utc)


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    age: Mapped[int]
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    predictions: Mapped[list["Prediction"]] = relationship(back_populates="patient")


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"))
    model_type: Mapped[str] = mapped_column(String(20))  # "diabetes" or "brain_tumor"
    result: Mapped[str] = mapped_column(String(50))
    confidence: Mapped[float]
    details: Mapped[str | None] = mapped_column(Text, nullable=True)  # input vitals as JSON
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    patient: Mapped["Patient"] = relationship(back_populates="predictions")