"""
URL configuration for Skin_Disease_Detection project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
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
from django.urls import path
from mainpage import views  # Import views from the mainpage app
from django.contrib.auth import views as auth_views  # Import for logout view
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),  # Admin site URL
    path('', views.mainpage, name='main'),  # Main page view
    path('loginpage/', views.loginpage, name='login'),  # Login page view
    path('signuppage/', views.signuppage, name='signup'),  # Signup page view
    path('profilepage/', views.profilepage, name='profile'),  # Profile page view
    path('logout/', auth_views.LogoutView.as_view(next_page='/'), name='logout'),  # Logout view
]

# Add static URL configuration for serving media files during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
