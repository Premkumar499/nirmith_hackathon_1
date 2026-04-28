"""
URL configuration for edna project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/6.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.http import FileResponse, HttpResponse
from django.views.generic.base import RedirectView
import os

def favicon_view(request):
    favicon_path = os.path.join(settings.BASE_DIR, 'static', 'favicon.svg')
    if os.path.exists(favicon_path):
        with open(favicon_path, 'rb') as f:
            return FileResponse(f, content_type='image/svg+xml')
    return HttpResponse(status=204)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('favicon.ico', favicon_view),
    path('', include('model.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
