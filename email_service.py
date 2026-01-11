"""
Email Service Module
Handles sending email notifications for inventory reorder alerts via Outlook
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import pandas as pd
from typing import List, Dict, Optional
import logging
from datetime import datetime

from config import (
    SMTP_SERVER, SMTP_PORT, EMAIL_SUBJECT_TEMPLATE, 
    EMAIL_BODY_TEMPLATE
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmailService:
    """
    Service for sending email notifications via Outlook SMTP
    """
    
    def __init__(self, sender_email: str, sender_password: str):
        """
        Initialize email service
        
        Args:
            sender_email: Sender's email address
            sender_password: Sender's email password or app password
        """
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.smtp_server = SMTP_SERVER
        self.smtp_port = SMTP_PORT
        
    def create_reorder_email(self, item_data: Dict, recipient_email: str) -> MIMEMultipart:
        """
        Create email message for reorder alert
        
        Args:
            item_data: Dictionary with item information
            recipient_email: Recipient's email address
            
        Returns:
            MIMEMultipart email message
        """
        # Create message
        message = MIMEMultipart()
        message['From'] = self.sender_email
        message['To'] = recipient_email
        message['Subject'] = EMAIL_SUBJECT_TEMPLATE.format(
            item_description=item_data.get('item_description', 'Unknown Item')
        )
        
        # Create email body
        body = EMAIL_BODY_TEMPLATE.format(
            item_code=item_data.get('item_code', 'N/A'),
            item_description=item_data.get('item_description', 'N/A'),
            equipment_name=item_data.get('equipment_name', 'N/A'),
            current_stock=f"{item_data.get('current_stock', 0):.2f}",
            reorder_level=f"{item_data.get('reorder_point', 0):.2f}",
            order_quantity=f"{item_data.get('recommended_order_quantity', 0):.2f}",
            forecasted_demand=f"{item_data.get('forecasted_30day_demand', 0):.2f}"
        )
        
        message.attach(MIMEText(body, 'plain'))
        
        return message
    
    def create_batch_reorder_email(self, items_data: List[Dict], 
                                   recipient_email: str) -> MIMEMultipart:
        """
        Create email message for multiple reorder alerts
        
        Args:
            items_data: List of dictionaries with item information
            recipient_email: Recipient's email address
            
        Returns:
            MIMEMultipart email message
        """
        # Create message
        message = MIMEMultipart()
        message['From'] = self.sender_email
        message['To'] = recipient_email
        message['Subject'] = f"Inventory Reorder Alert: {len(items_data)} Items Require Attention"
        
        # Create email body
        body = f"""
Dear Inventory Manager,

This is an automated alert for {len(items_data)} items that require reordering:

"""
        
        for i, item in enumerate(items_data, 1):
            body += f"""
{i}. Item: {item.get('item_description', 'N/A')}
   Item Code: {item.get('item_code', 'N/A')}
   Equipment: {item.get('equipment_name', 'N/A')}
   Current Stock: {item.get('current_stock', 0):.2f}
   Reorder Level: {item.get('reorder_point', 0):.2f}
   Recommended Order: {item.get('recommended_order_quantity', 0):.2f}
   Days Until Stockout: {item.get('days_until_stockout', 0):.1f}
   
"""
        
        body += """
Action Required: Please review and initiate purchase orders for the above items.

Best regards,
AI Inventory Management System
"""
        
        message.attach(MIMEText(body, 'plain'))
        
        return message
    
    def attach_csv_report(self, message: MIMEMultipart, df: pd.DataFrame, 
                         filename: str = 'reorder_report.csv'):
        """
        Attach CSV report to email
        
        Args:
            message: Email message
            df: DataFrame to attach
            filename: Name of the CSV file
        """
        try:
            # Convert DataFrame to CSV
            csv_data = df.to_csv(index=False)
            
            # Create attachment
            attachment = MIMEBase('application', 'octet-stream')
            attachment.set_payload(csv_data.encode('utf-8'))
            encoders.encode_base64(attachment)
            attachment.add_header('Content-Disposition', f'attachment; filename={filename}')
            
            message.attach(attachment)
            
            logger.info(f"CSV report attached: {filename}")
            
        except Exception as e:
            logger.error(f"Error attaching CSV report: {str(e)}")
    
    def send_email(self, message: MIMEMultipart) -> bool:
        """
        Send email via SMTP
        
        Args:
            message: Email message to send
            
        Returns:
            Boolean indicating success
        """
        try:
            # Connect to SMTP server
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            
            # Login
            server.login(self.sender_email, self.sender_password)
            
            # Send email
            server.send_message(message)
            
            # Close connection
            server.quit()
            
            logger.info(f"Email sent successfully to {message['To']}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {str(e)}")
            return False
    
    def send_reorder_alert(self, item_data: Dict, recipient_email: str) -> bool:
        """
        Send reorder alert for a single item
        
        Args:
            item_data: Dictionary with item information
            recipient_email: Recipient's email address
            
        Returns:
            Boolean indicating success
        """
        try:
            message = self.create_reorder_email(item_data, recipient_email)
            return self.send_email(message)
        except Exception as e:
            logger.error(f"Error sending reorder alert: {str(e)}")
            return False
    
    def send_batch_reorder_alert(self, items_data: List[Dict], 
                                 recipient_email: str,
                                 attach_report: bool = True) -> bool:
        """
        Send batch reorder alert for multiple items
        
        Args:
            items_data: List of dictionaries with item information
            recipient_email: Recipient's email address
            attach_report: Whether to attach CSV report
            
        Returns:
            Boolean indicating success
        """
        try:
            message = self.create_batch_reorder_email(items_data, recipient_email)
            
            # Attach CSV report if requested
            if attach_report and items_data:
                df = pd.DataFrame(items_data)
                self.attach_csv_report(message, df)
            
            return self.send_email(message)
            
        except Exception as e:
            logger.error(f"Error sending batch reorder alert: {str(e)}")
            return False
    
    def test_connection(self) -> Dict:
        """
        Test SMTP connection and credentials
        
        Returns:
            Dictionary with test results
        """
        try:
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            server.quit()
            
            return {
                'success': True,
                'message': 'Connection successful'
            }
            
        except smtplib.SMTPAuthenticationError:
            return {
                'success': False,
                'message': 'Authentication failed. Please check email and password.'
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Connection failed: {str(e)}'
            }


def send_reorder_notifications(recommendations_df: pd.DataFrame, 
                               sender_email: str, 
                               sender_password: str,
                               recipient_email: str,
                               batch_mode: bool = True) -> Dict:
    """
    Send reorder notifications for items that need reordering
    
    Args:
        recommendations_df: DataFrame with inventory recommendations
        sender_email: Sender's email address
        sender_password: Sender's email password
        recipient_email: Recipient's email address
        batch_mode: Send one email with all items or individual emails
        
    Returns:
        Dictionary with sending results
    """
    logger.info("Sending reorder notifications...")
    
    # Filter items that need reorder
    items_to_reorder = recommendations_df[recommendations_df['needs_reorder'] == True]
    
    if items_to_reorder.empty:
        return {
            'success': True,
            'message': 'No items require reordering',
            'items_sent': 0
        }
    
    # Initialize email service
    email_service = EmailService(sender_email, sender_password)
    
    # Test connection first
    test_result = email_service.test_connection()
    if not test_result['success']:
        return {
            'success': False,
            'message': test_result['message'],
            'items_sent': 0
        }
    
    try:
        if batch_mode:
            # Send one email with all items
            items_data = items_to_reorder.to_dict('records')
            success = email_service.send_batch_reorder_alert(
                items_data, 
                recipient_email,
                attach_report=True
            )
            
            return {
                'success': success,
                'message': f'Batch email sent with {len(items_data)} items' if success else 'Failed to send batch email',
                'items_sent': len(items_data) if success else 0
            }
        else:
            # Send individual emails
            sent_count = 0
            for _, item in items_to_reorder.iterrows():
                item_data = item.to_dict()
                if email_service.send_reorder_alert(item_data, recipient_email):
                    sent_count += 1
            
            return {
                'success': sent_count > 0,
                'message': f'Sent {sent_count} out of {len(items_to_reorder)} emails',
                'items_sent': sent_count
            }
            
    except Exception as e:
        logger.error(f"Error sending notifications: {str(e)}")
        return {
            'success': False,
            'message': f'Error: {str(e)}',
            'items_sent': 0
        }
