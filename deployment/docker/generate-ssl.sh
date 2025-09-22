#!/bin/bash

# Generate self-signed SSL certificates for VecLite Docker deployment
# For development and testing purposes only

set -e

# Create SSL directory
mkdir -p nginx/ssl

# Configuration for the certificate
COUNTRY="US"
STATE="California"
CITY="San Francisco"
ORG="VecLite"
ORG_UNIT="Development"
COMMON_NAME="veclite.local"
EMAIL="admin@veclite.local"

echo "Generating self-signed SSL certificate for VecLite..."
echo "Common Name: $COMMON_NAME"

# Generate private key
openssl genrsa -out nginx/ssl/key.pem 2048

# Generate certificate signing request
openssl req -new -key nginx/ssl/key.pem -out nginx/ssl/cert.csr -subj "/C=$COUNTRY/ST=$STATE/L=$CITY/O=$ORG/OU=$ORG_UNIT/CN=$COMMON_NAME/emailAddress=$EMAIL"

# Generate self-signed certificate
openssl x509 -req -days 365 -in nginx/ssl/cert.csr -signkey nginx/ssl/key.pem -out nginx/ssl/cert.pem

# Clean up CSR
rm nginx/ssl/cert.csr

# Set appropriate permissions
chmod 600 nginx/ssl/key.pem
chmod 644 nginx/ssl/cert.pem

echo "SSL certificate generated successfully!"
echo "Certificate: nginx/ssl/cert.pem"
echo "Private key: nginx/ssl/key.pem"
echo ""
echo "To use HTTPS, add '127.0.0.1 veclite.local' to your /etc/hosts file"
echo "and access VecLite at https://veclite.local"
echo ""
echo "Note: This is a self-signed certificate for development only."
echo "For production, use certificates from a trusted CA."