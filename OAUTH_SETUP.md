# Google OAuth Setup Guide for Vigi-Drive

This guide will walk you through setting up Google OAuth authentication for your Vigi-Drive application.

## Prerequisites
- A Google account (Gmail)
- About 5-10 minutes

## Step 1: Access Google Cloud Console

1. Open your browser and go to: **https://console.cloud.google.com**
2. Sign in with your Google account if prompted

## Step 2: Create a New Project

1. At the top of the page, click the **project dropdown** (next to "Google Cloud")
2. Click **"NEW PROJECT"** in the top right
3. Enter project name: **`Vigi-Drive`**
4. Leave organization as-is
5. Click **"CREATE"**
6. Wait for the project to be created (takes a few seconds)
7. Select your new project from the dropdown

## Step 3: Enable Google Identity Services

1. In the left sidebar, navigate to: **"APIs & Services"** → **"Enabled APIs & Services"**
2. Click **"+ ENABLE APIS AND SERVICES"** at the top
3. Search for: **"Google+ API"** or **"Google Identity"**
4. Click on it, then click **"ENABLE"**
5. Wait for it to enable

## Step 4: Configure OAuth Consent Screen

1. In the left sidebar, go to: **"APIs & Services"** → **"OAuth consent screen"**
2. Select **"External"** user type (unless you have a Google Workspace)
3. Click **"CREATE"**

### Fill in the App Information:
- **App name:** `Vigi-Drive Drowsiness Detection`
- **User support email:** Select your email from dropdown
- **App logo:** (Optional - skip for now)
- **Application home page:** `http://localhost:5001`
- **Application privacy policy link:** (Optional - can skip)
- **Application terms of service link:** (Optional - can skip)
- **Authorized domains:** Leave empty for now
- **Developer contact information:** Enter your email

4. Click **"SAVE AND CONTINUE"**

### Scopes:
5. Click **"ADD OR REMOVE SCOPES"**
6. Select these scopes:
   - `.../auth/userinfo.email`
   - `.../auth/userinfo.profile`
   - `openid`
7. Click **"UPDATE"**
8. Click **"SAVE AND CONTINUE"**

### Test Users (Important for Development):
9. Click **"+ ADD USERS"**
10. Enter your Gmail address
11. Click **"ADD"**
12. Click **"SAVE AND CONTINUE"**

13. Review and click **"BACK TO DASHBOARD"**

## Step 5: Create OAuth Credentials

1. In the left sidebar, go to: **"APIs & Services"** → **"Credentials"**
2. Click **"+ CREATE CREDENTIALS"** at the top
3. Select **"OAuth client ID"**

### Configure the OAuth Client:
4. **Application type:** Select **"Web application"**
5. **Name:** `Vigi-Drive Web Client`

### Authorized JavaScript origins:
6. Click **"+ ADD URI"**
   - Add: `http://localhost:5001`
7. Click **"+ ADD URI"** again
   - Add: `http://127.0.0.1:5001`

### Authorized redirect URIs:
8. Click **"+ ADD URI"**
   - Add: `http://localhost:5001/auth/google/callback`
9. Click **"+ ADD URI"** again
   - Add: `http://127.0.0.1:5001/auth/google/callback`

10. Click **"CREATE"**

## Step 6: Save Your Credentials

A popup will appear with your credentials:

```
Your Client ID
------------------
[COPY THIS - looks like: 123456789-abcdefg.apps.googleusercontent.com]

Your Client Secret
------------------
[COPY THIS - looks like: GOCSPX-abc123def456]
```

11. **Click the COPY button** for Client ID and save it somewhere safe
12. **Click the COPY button** for Client Secret and save it somewhere safe
13. Click **"OK"**

> ⚠️ **IMPORTANT:** Keep these credentials secure. Don't share them publicly or commit them to git.

## Step 7: Ready to Configure Your App

Once you have both credentials copied, let me know and I'll help you:
1. Create a `.env` file with your credentials
2. Update the application to use them
3. Test the login flow

---

## Troubleshooting

### "Access blocked: Vigi-Drive has not completed the Google verification process"
- This is normal for development
- Make sure you added your email as a test user in Step 4
- Only test users can log in during development

### Can't find the OAuth consent screen
- Make sure you selected your project in the top dropdown
- Navigate to: APIs & Services → OAuth consent screen

### Redirect URI mismatch error
- Double-check the URIs match exactly:
  - `http://localhost:5001/auth/google/callback`
  - `http://127.0.0.1:5001/auth/google/callback`
- No trailing slashes

---

## Next Steps

After you have your credentials, paste them here and I'll:
1. Set up your `.env` file
2. Configure the application
3. Test the Google login

**Reply with:**
```
Client ID: [paste here]
Client Secret: [paste here]
```
