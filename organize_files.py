#!/usr/bin/env python3
"""
YouTube MLOps Project File Structure Organizer
This script creates the correct file structure and identifies files that need updates.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Set

class MLOpsFileOrganizer:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.required_structure = self._get_required_structure()
        self.files_to_update = self._get_files_to_update()
        self.files_to_remove = self._get_files_to_remove()
        
    def _get_required_structure(self) -> Dict[str, List[str]]:
        """Define the required project structure"""
        return {
            # Root level files
            "": [
                ".env.template",
                ".gitignore", 
                "Dockerfile",
                "README.md",
                "dashboard.py",
                "docker-compose.yml",
                "dvc.yaml",
                "params.yaml", 
                "requirements.txt",
                "run_pipeline.sh",
                "setup.sh"
            ],
            
            # Data directories
            "data": [],
            "data/raw": [".gitkeep"],
            "data/processed": [".gitkeep"],
            "data/external": [],
            
            # Models directory
            "models": [".gitkeep"],
            
            # Logs directory  
            "logs": [".gitkeep"],
            
            # Metrics directory
            "metrics": [".gitkeep"],
            
            # Source code structure
            "src": ["__init__.py"],
            "src/data": ["__init__.py", "data_collector.py", "feature_engineering.py"],
            "src/models": ["__init__.py", "train.py", "evaluate.py"],
            "src/api": ["__init__.py", "main.py", "schemas.py"],
            "src/monitoring": ["__init__.py", "metrics.py"],
            "src/utils": ["__init__.py", "config.py", "helpers.py"],
            
            # Tests structure
            "tests": ["__init__.py"],
            "tests/test_data": ["__init__.py"],
            "tests/test_models": ["__init__.py", "test_model.py"],
            "tests/test_api": ["__init__.py", "test_endpoints.py"],
            
            # Deployment structure
            "deployment": [],
            "deployment/docker": ["Dockerfile.api", "Dockerfile.training", "docker-compose.yml", "prometheus.yml"],
            "deployment/kubernetes": ["api-deployment.yaml"],
            
            # Airflow structure
            "airflow": [],
            "airflow/dags": ["training_pipeline.py"],
            
            # Monitoring structure  
            "monitoring": [],
            "monitoring/grafana": [],
            "monitoring/grafana/dashboards": ["model_monitoring.json", "rules.yml"],
            "monitoring/prometheus": [],
            
            # GitHub Actions
            ".github": [],
            ".github/workflows": ["mlops-pipeline.yml"],
            
            # Notebooks (optional)
            "notebooks": [
                "01_data_exploration.ipynb",
                "02_feature_engineering.ipynb", 
                "03_model_development.ipynb",
                "04_model_evaluation.ipynb"
            ]
        }
    
    def _get_files_to_update(self) -> Set[str]:
        """Files that need content updates"""
        return {
            "src/data/data_collector.py",
            "src/data/feature_engineering.py", 
            "src/models/train.py",
            "src/api/main.py",
            "requirements.txt",
            "README.md",
            "Dockerfile"
        }
    
    def _get_files_to_remove(self) -> Set[str]:
        """Files that should be removed (not in required structure)"""
        return {
            "src/data/data_processor.py",  # Empty file, not needed
            "src/models/model_utils.py",   # Empty file, not needed  
            "src/models/predict.py",       # Empty file, not needed
            "setup.py",                    # Empty file, not needed
            "mlflow.yaml",                 # Empty file, not needed
            "deployment/docker/schemas.py", # Duplicate of src/api/schemas.py
        }
    
    def create_directory_structure(self):
        """Create all required directories"""
        print("🏗️  Creating directory structure...")
        
        for dir_path in self.required_structure.keys():
            if dir_path:  # Skip empty string (root)
                full_path = self.project_root / dir_path
                full_path.mkdir(parents=True, exist_ok=True)
                print(f"✅ Created/verified: {dir_path}/")
    
    def create_missing_files(self):
        """Create missing files with appropriate content"""
        print("\n📄 Creating missing files...")
        
        for dir_path, files in self.required_structure.items():
            for file_name in files:
                if dir_path:
                    file_path = self.project_root / dir_path / file_name
                else:
                    file_path = self.project_root / file_name
                
                if not file_path.exists():
                    self._create_file_with_content(file_path)
                    print(f"✅ Created: {file_path}")
                else:
                    print(f"⏭️  Exists: {file_path}")
    
    def _create_file_with_content(self, file_path: Path):
        """Create file with appropriate default content"""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if file_path.name == "__init__.py":
            file_path.write_text("# Python package initialization\n")
        
        elif file_path.name == ".gitkeep":
            file_path.write_text("# Keep this directory in git\n")
        
        elif file_path.suffix == ".sh":
            file_path.write_text("#!/bin/bash\n# Shell script placeholder\necho 'Script placeholder'\n")
            file_path.chmod(0o755)  # Make executable
        
        elif file_path.suffix == ".py" and "test_" in file_path.name:
            file_path.write_text("# Test file placeholder\nimport pytest\n\ndef test_placeholder():\n    assert True\n")
        
        elif file_path.suffix == ".py":
            file_path.write_text("# Python file placeholder\npass\n")
        
        elif file_path.suffix == ".yml" or file_path.suffix == ".yaml":
            file_path.write_text("# YAML configuration placeholder\n")
        
        elif file_path.suffix == ".json":
            file_path.write_text('{\n  "placeholder": "configuration"\n}\n')
        
        elif file_path.suffix == ".md":
            file_path.write_text(f"# {file_path.stem.replace('_', ' ').title()}\n\nPlaceholder content.\n")
        
        else:
            file_path.write_text("# Placeholder file\n")
    
    def remove_unwanted_files(self):
        """Remove files that are not needed"""
        print("\n🗑️  Removing unwanted files...")
        
        for file_path in self.files_to_remove:
            full_path = self.project_root / file_path
            if full_path.exists():
                if full_path.is_file():
                    full_path.unlink()
                    print(f"🗑️  Removed file: {file_path}")
                elif full_path.is_dir():
                    shutil.rmtree(full_path)
                    print(f"🗑️  Removed directory: {file_path}")
            else:
                print(f"⏭️  Already removed: {file_path}")
    
    def scan_existing_files(self) -> Dict[str, List[str]]:
        """Scan current directory structure"""
        print("\n🔍 Scanning current files...")
        
        current_structure = {}
        
        for root, dirs, files in os.walk(self.project_root):
            # Skip hidden directories and __pycache__
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            rel_path = os.path.relpath(root, self.project_root)
            if rel_path == ".":
                rel_path = ""
            
            current_structure[rel_path] = files
        
        return current_structure
    
    def generate_file_status_report(self):
        """Generate a report of file status"""
        print("\n📊 File Status Report")
        print("=" * 60)
        
        # Files that need manual updates
        print("\n🔄 FILES REQUIRING MANUAL UPDATES:")
        for file_path in sorted(self.files_to_update):
            full_path = self.project_root / file_path
            status = "EXISTS" if full_path.exists() else "MISSING"
            print(f"  📝 {file_path} - {status}")
        
        # New files to create
        print("\n🆕 NEW FILES TO CREATE:")
        new_files = ["setup.sh", "run_pipeline.sh", "dashboard.py", "docker-compose.yml"]
        for file_name in new_files:
            full_path = self.project_root / file_name
            status = "EXISTS" if full_path.exists() else "WILL CREATE"
            print(f"  ✨ {file_name} - {status}")
        
        # Files removed
        print("\n🗑️  UNWANTED FILES (REMOVED):")
        for file_path in sorted(self.files_to_remove):
            print(f"  ❌ {file_path}")
    
    def create_gitignore(self):
        """Create proper .gitignore file"""
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/

# Environment variables
.env

# Models and data
models/*.joblib
models/*.pkl
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Logs
logs/
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# MLflow
mlruns/
mlartifacts/

# Jupyter
.ipynb_checkpoints/

# Testing
.coverage
htmlcov/
.pytest_cache/

# Temporary files
*.tmp
*.bak
"""
        gitignore_path = self.project_root / ".gitignore"
        if not gitignore_path.exists():
            gitignore_path.write_text(gitignore_content)
            print("✅ Created .gitignore")
    
    def run_organization(self):
        """Run the complete file organization process"""
        print("🚀 Starting MLOps Project File Organization")
        print("=" * 50)
        
        # Step 1: Remove unwanted files first
        self.remove_unwanted_files()
        
        # Step 2: Create directory structure
        self.create_directory_structure()
        
        # Step 3: Create missing files
        self.create_missing_files()
        
        # Step 4: Create .gitignore
        self.create_gitignore()
        
        # Step 5: Generate status report
        self.generate_file_status_report()
        
        print("\n✅ File organization completed!")
        print("\n📋 Next Steps:")
        print("1. Update the files marked as 'REQUIRING MANUAL UPDATES'")
        print("2. Run: chmod +x setup.sh run_pipeline.sh")
        print("3. Execute: ./setup.sh")
        print("4. Test: ./run_pipeline.sh")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Organize MLOps project file structure")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    
    args = parser.parse_args()
    
    organizer = MLOpsFileOrganizer(args.project_root)
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")
        organizer.scan_existing_files()
        organizer.generate_file_status_report()
    else:
        organizer.run_organization()

if __name__ == "__main__":
    main()