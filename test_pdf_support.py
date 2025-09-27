#!/usr/bin/env python3
"""
Test script to verify PDF support for bank_handbook.pdf
"""

import os
import sys
from pathlib import Path

def test_pdf_libraries():
    """Test available PDF libraries and their capabilities."""
    print("🔍 Testing PDF Library Support\n")
    
    libraries = []
    
    # Test PyPDF2
    try:
        from PyPDF2 import PdfReader
        libraries.append(("PyPDF2", PdfReader, "✅ Basic PDF support"))
    except ImportError:
        print("❌ PyPDF2 not available")
    
    # Test pypdf
    try:
        from pypdf import PdfReader
        libraries.append(("pypdf", PdfReader, "✅ Modern PDF support"))
    except ImportError:
        print("❌ pypdf not available")
    
    # Test pdfplumber
    try:
        import pdfplumber
        libraries.append(("pdfplumber", pdfplumber, "✅ Advanced layout support"))
    except ImportError:
        print("❌ pdfplumber not available")
    
    # Test PyMuPDF
    try:
        import fitz
        libraries.append(("PyMuPDF", fitz, "✅ High performance PDF"))
    except ImportError:
        print("❌ PyMuPDF not available")
    
    if not libraries:
        print("\n⚠️  No PDF libraries found!")
        print("Install with: poetry install --extras pdf-best")
        return False
    
    print(f"\n✅ Found {len(libraries)} PDF libraries:")
    for name, lib, desc in libraries:
        print(f"   • {name}: {desc}")
    
    return True


def test_bank_handbook():
    """Test specific extraction from bank_handbook.pdf."""
    pdf_path = Path("docs/bank_handbook.pdf")
    
    if not pdf_path.exists():
        print(f"❌ Bank handbook not found at {pdf_path}")
        return False
    
    print(f"\n📄 Testing {pdf_path} ({pdf_path.stat().st_size} bytes)")
    
    # Test with available libraries
    success = False
    
    # Try PyPDF2/pypdf
    for lib_name in ["PyPDF2", "pypdf"]:
        try:
            if lib_name == "PyPDF2":
                from PyPDF2 import PdfReader
            else:
                from pypdf import PdfReader
            
            with open(pdf_path, 'rb') as f:
                reader = PdfReader(f)
                pages = len(reader.pages)
                sample_text = reader.pages[0].extract_text()[:200]
                
            print(f"✅ {lib_name}: {pages} pages, sample: {sample_text[:100]}...")
            success = True
            break
        except ImportError:
            continue
        except Exception as e:
            print(f"❌ {lib_name} failed: {e}")
    
    # Try pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            pages = len(pdf.pages)
            sample_text = pdf.pages[0].extract_text()[:200]
        print(f"✅ pdfplumber: {pages} pages, sample: {sample_text[:100]}...")
        success = True
    except ImportError:
        pass
    except Exception as e:
        print(f"❌ pdfplumber failed: {e}")
    
    # Try PyMuPDF
    try:
        import fitz
        doc = fitz.open(pdf_path)
        pages = doc.page_count
        sample_text = doc[0].get_text()[:200]
        doc.close()
        print(f"✅ PyMuPDF: {pages} pages, sample: {sample_text[:100]}...")
        success = True
    except ImportError:
        pass
    except Exception as e:
        print(f"❌ PyMuPDF failed: {e}")
    
    return success


def test_chat_integration():
    """Test integration with chat.py."""
    print("\n🔗 Testing Chat Integration")
    
    try:
        from scripts.chat import load_documents, PDF_READERS, HAS_PDF
        
        print(f"PDF support: {HAS_PDF}")
        print(f"Available readers: {list(PDF_READERS.keys())}")
        
        # Test document loading
        documents = load_documents("docs")
        pdf_docs = [doc for doc in documents if doc.metadata.get('file_type') == '.pdf']
        
        print(f"✅ Loaded {len(documents)} documents total")
        print(f"✅ Found {len(pdf_docs)} PDF documents")
        
        if pdf_docs:
            bank_doc = next((doc for doc in pdf_docs if 'bank_handbook' in doc.metadata['file_name']), None)
            if bank_doc:
                print(f"✅ Bank handbook loaded: {len(bank_doc.page_content)} characters")
                print(f"   Sample: {bank_doc.page_content[:150]}...")
                return True
            else:
                print("❌ Bank handbook not found in loaded documents")
        
        return False
        
    except Exception as e:
        print(f"❌ Chat integration failed: {e}")
        return False


def main():
    """Run all tests."""
    print("📋 PDF Support Test for Bank Handbook\n")
    
    results = []
    
    # Test 1: Library availability
    results.append(("PDF Libraries", test_pdf_libraries()))
    
    # Test 2: Bank handbook extraction
    results.append(("Bank Handbook", test_bank_handbook()))
    
    # Test 3: Chat integration
    results.append(("Chat Integration", test_chat_integration()))
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST RESULTS:")
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Your bank handbook is ready to use.")
    else:
        print("\n⚠️  Some tests failed. Install PDF libraries:")
        print("   poetry install --extras pdf-best")
        print("   # or")
        print("   pip install pymupdf pdfplumber")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
