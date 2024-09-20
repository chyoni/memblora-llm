from django.urls import path

from . import views
from .views import EmbeddingView, QueryView

urlpatterns = [
    path("embedding", EmbeddingView.as_view(), name="file-embedding"),
    path("query", QueryView.as_view(), name="query"),
]