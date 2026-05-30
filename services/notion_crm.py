import os
from notion_client import AsyncClient


class NotionCRM:
    def __init__(self):
        self.secret = os.getenv("NOTION_SECRET_KEY")
        self.database_id = os.getenv("NOTION_DATABASE_ID")
        # Only initialize if the key exists to prevent local crashes
        self.client = AsyncClient(auth=self.secret) if self.secret else None

    async def create_lead(self, inquiry):
        if not self.client or not self.database_id:
            print("Notion credentials missing. Skipping CRM sync.")
            return False

        try:
            await self.client.pages.create(
                parent={"database_id": self.database_id},
                properties={
                    "Agency Name": {
                        "title": [{"text": {"content": inquiry.corporate_entity}}]
                    },
                    "Contact": {
                        "email": inquiry.preferred_contact
                    },
                    "Volume": {
                        "rich_text": [{"text": {"content": inquiry.portfolio_volume}}]
                    },
                    "Region": {
                        "rich_text": [{"text": {"content": inquiry.operational_region}}]
                    },
                    "Status": {
                        "status": {"name": "New Lead"}  # Assuming you used the 'Status' property type
                    }
                },
                # This injects the full message into the body of the Notion page
                children=[
                    {
                        "object": "block",
                        "type": "heading_3",
                        "heading_3": {"rich_text": [{"text": {"content": "Inquiry Details"}}]},
                    },
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {"rich_text": [{"text": {"content": inquiry.inquiry_details}}]},
                    },
                ],
            )
            return True
        except Exception as e:
            print(f"Notion sync failed: {e}")
            return False


# Export an instance to use in your routes
notion_crm = NotionCRM()
