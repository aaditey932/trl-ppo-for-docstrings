import os
import webbrowser
import time
from dotenv import load_dotenv
import httpx
import msal

load_dotenv()


# ========================GET ACCESS TOKEN=======================================
def is_browser_available():
    try:
        # Attempt to open a dummy URL
        browser = webbrowser.get()
        return browser.name is not None
    except webbrowser.Error:
        print("Error: Browser check failed")
        return False

def get_access_token_client_credential(
    clientid, tenant_id="common", scopes=None, secret=None
):
    """
    Checks if the access token is blank or expired.
    If expired, requests a new access token.

    :param access_token: The OAuth access token
    :param expires_at: Token expiration timestamp
    :return: A valid access token
    """
    if scopes is None:
        scopes = ["User.Read"]
    AUTHORITY = "https://login.microsoftonline.com/" + tenant_id
    app = msal.ConfidentialClientApplication(
        clientid, authority=AUTHORITY, client_credential=secret
    )
    token_response = app.acquire_token_for_client(scopes=scopes)
    if "access_token" in token_response:
        new_access_token = token_response["access_token"]
        expires_in = token_response.get("expires_in", 0)
        new_expires_at = time.time() + expires_in
        return new_access_token, new_expires_at

    return None, None

def get_access_token_interactive(
    client_id, tenant_id="common", scopes=None, timeout=15
):
    """
    Returns only the access token using MSAL's Interactive Flow.

    Args:
        client_id (str): Azure AD application (client) ID.
        tenant_id (str): Directory (tenant) ID or 'common'.
        scopes (list): List of scopes to request (e.g., ['User.Read']).

    Returns:
        str or None: Access token if successful, otherwise None.
    """
    if scopes is None:
        scopes = ["User.Read"]
    AUTHORITY = "https://login.microsoftonline.com/" + tenant_id
    app = msal.PublicClientApplication(client_id=client_id, authority=AUTHORITY)
    token_response = app.acquire_token_interactive(scopes=scopes, timeout=timeout)
    if "access_token" in token_response:
        new_access_token = token_response["access_token"]
        expires_in = token_response.get("expires_in", 0)
        new_expires_at = time.time() + expires_in
        return new_access_token, new_expires_at

    return None, None

def get_access_token_device_flow(client_id, tenant_id="common", scopes=None):
    """
    Returns only the access token using MSAL's Device Code Flow.

    Args:
        client_id (str): Azure AD application (client) ID.
        tenant_id (str): Directory (tenant) ID or 'common'.
        scopes (list): List of scopes to request (e.g., ['User.Read']).

    Returns:
        str or None: Access token if successful, otherwise None.
    """
    if scopes is None:
        scopes = ["User.Read"]
    AUTHORITY = "https://login.microsoftonline.com/" + tenant_id
    app = msal.PublicClientApplication(client_id=client_id, authority=AUTHORITY)
    flow = app.initiate_device_flow(scopes=scopes)
    if "user_code" not in flow:
        raise ValueError("Failed to initiate device flow.")
    token_response = app.acquire_token_by_device_flow(flow)
    if "access_token" in token_response:
        new_access_token = token_response["access_token"]
        expires_in = token_response.get("expires_in", 0)
        new_expires_at = time.time() + expires_in
        return new_access_token, new_expires_at
    return None, None

def get_access_token(
    access_token, expires_at, clientid, tenant_id="common", scopes=None, secret=None
):
    """
    Checks if the access token is blank or expired.
    If expired, requests a new access token.

    :param access_token: The OAuth access token
    :param expires_at: Token expiration timestamp
    :return: A valid access token
    """
    if not access_token or time.time() >= expires_at:
        new_access_token = None
        if secret is not None:
            new_access_token, new_expires_at = get_access_token_client_credential(
                clientid, tenant_id, scopes, secret
            )
            if new_access_token is None:

                return None, None
            else:
                return new_access_token, new_expires_at
        if is_browser_available():
            new_access_token, new_expires_at = get_access_token_interactive(
                clientid, tenant_id, scopes
            )
        else:
            print("No browser available; try device code")

        if new_access_token is None:
            print(
                "Error: Failed to get token interactively; trying device code flow"
            )
            new_access_token, new_expires_at = get_access_token_device_flow(
                clientid, tenant_id, scopes
            )
            if new_access_token is None:
                print("Error: Failed to get token using all methods")
                return None, None
            else:
                return new_access_token, new_expires_at
        else:
            return new_access_token, new_expires_at

    return access_token, expires_at


# =======================vLLM HOSTED LLM OUTPUT GENERATION========================================
class LLMClient:
    """
    Handles authentication and LLM API calls for GPT and Mistral models.
    """

    def __init__(self):
        self.scope = [os.environ["KUBEFLOW_OIDC_SCOPE"]]
        self.access_token, self.expires_at = get_access_token(
            access_token=None,
            expires_at=0,
            clientid=os.environ["CCI_AIL_DOCUMENT_HIERARCHY_CLIENTID"],
            tenant_id=os.environ["PRGX_AZURE_TENANT_ID"],
            scopes=self.scope,
            secret=os.environ.get("CCI_AIL_DOCUMENT_HIERARCHY_SECRET", None),
        )

    def _refresh_token_if_needed(self):
        if time.time() > self.expires_at - 60:  # Refresh 1 min before expiry
            self.access_token, self.expires_at = get_access_token(
                access_token=None,
                expires_at=0,
                clientid=os.environ["CCI_AIL_DOCUMENT_HIERARCHY_CLIENTID"],
                tenant_id=os.environ["PRGX_AZURE_TENANT_ID"],
                scopes=self.scope,
                secret=os.environ.get("CCI_AIL_DOCUMENT_HIERARCHY_SECRET", None),
            )

    def _generate_llm_response(
        self,
        model_name: str,
        model_url: str,
        messages: list,
        verify_cert: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """
        Send a request to a specified LLM (GPT or Mistral) via vLLM API and return the generated response.
        """

        # Validate model_name
        allowed_models = {"GPT", "MISTRAL"}
        if model_name.upper() not in allowed_models:
            raise ValueError(
                f"Invalid model_name '{model_name}'. Allowed values: {allowed_models}"
            )

        # Refresh token if expired
        self._refresh_token_if_needed()

        hdr = {"Authorization": f"Bearer {self.access_token}"}

        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        start_time = time.time()
        status_code = -1
        retry = 0
        response = None
        while retry < 3:
            try:
                response = httpx.post(
                    url=model_url,
                    headers=hdr,
                    json=payload,
                    verify=verify_cert,
                    timeout=httpx.Timeout(
                        connect=15.0,
                        read=300.0,
                        write=60.0,
                        pool=15.0,
                    ),
                )
                status_code = response.status_code
                if status_code == 200:
                    break

                print(
                    f"Received error status {status_code} on attempt {retry+1}"
                )
            except Exception as e:
                print(
                    f"Exception during {model_name} request on attempt {retry+1}: {e}"
                )
            retry += 1
            time.sleep(60)

        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000

        if response and response.status_code == 200:
            print(
                f"Successfully generated {model_name} response\nResponse Time: {elapsed_time:.2f} ms"
            )
            response_json = response.json()
            actual_response = response_json["choices"][0]["message"]["content"]
            return actual_response
        else:
            error_msg = (
                f"{model_name} API failed after {retry} retries. Status: {status_code}"
            )
            print(
                f"{error_msg}. Response: {response.text if response else 'No response'}"
            )
            raise Exception(error_msg)

