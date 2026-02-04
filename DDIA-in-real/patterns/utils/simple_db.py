"""
Simple Database Simulator for Pattern Demonstrations
A basic in-memory database with transaction support

This utility is used by pattern implementations (e.g., Transactional Writer)
to demonstrate transactional behavior in a simple, self-contained way.
"""

import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import time


class TransactionStatus(Enum):
    ACTIVE = "active"
    COMMITTED = "committed"
    ABORTED = "aborted"


@dataclass
class Transaction:
    id: int
    status: TransactionStatus
    changes: Dict[str, Any]  # table_name -> {key -> value}
    start_time: float


class SimpleDB:
    def __init__(self):
        self.data: Dict[str, Dict[str, Any]] = {}  # table_name -> {key -> value}
        self.transactions: Dict[int, Transaction] = {}
        self.next_transaction_id = 1
        self.lock = threading.Lock()
    
    def create_table(self, table_name: str) -> None:
        """Create a new table"""
        with self.lock:
            if table_name not in self.data:
                self.data[table_name] = {}
                print(f"✅ Table '{table_name}' created")
            else:
                print(f"⚠️  Table '{table_name}' already exists")
    
    def insert(self, table_name: str, key: str, value: Any, transaction_id: Optional[int] = None) -> bool:
        """Insert or update a record"""
        if transaction_id is None:
            # Direct operation (no transaction)
            with self.lock:
                if table_name not in self.data:
                    self.data[table_name] = {}
                self.data[table_name][key] = value
                print(f"✅ Inserted {key} = {value} into {table_name}")
                return True
        else:
            # Transaction operation
            if transaction_id not in self.transactions:
                print(f"❌ Transaction {transaction_id} not found")
                return False
            
            transaction = self.transactions[transaction_id]
            if transaction.status != TransactionStatus.ACTIVE:
                print(f"❌ Transaction {transaction_id} is not active")
                return False
            
            if table_name not in transaction.changes:
                transaction.changes[table_name] = {}
            transaction.changes[table_name][key] = value
            print(f"📝 Transaction {transaction_id}: Staged insert {key} = {value} into {table_name}")
            return True
    
    def select(self, table_name: str, key: str, transaction_id: Optional[int] = None) -> Optional[Any]:
        """Select a record"""
        if transaction_id is None:
            # Direct operation
            with self.lock:
                if table_name in self.data and key in self.data[table_name]:
                    value = self.data[table_name][key]
                    print(f"✅ Selected {key} = {value} from {table_name}")
                    return value
                else:
                    print(f"❌ Key {key} not found in {table_name}")
                    return None
        else:
            # Transaction operation - read from staged changes first, then from committed data
            if transaction_id not in self.transactions:
                print(f"❌ Transaction {transaction_id} not found")
                return None
            
            transaction = self.transactions[transaction_id]
            if transaction.status != TransactionStatus.ACTIVE:
                print(f"❌ Transaction {transaction_id} is not active")
                return None
            
            # Check staged changes first
            if table_name in transaction.changes and key in transaction.changes[table_name]:
                value = transaction.changes[table_name][key]
                print(f"📝 Transaction {transaction_id}: Read staged {key} = {value} from {table_name}")
                return value
            
            # Check committed data
            with self.lock:
                if table_name in self.data and key in self.data[table_name]:
                    value = self.data[table_name][key]
                    print(f"📝 Transaction {transaction_id}: Read committed {key} = {value} from {table_name}")
                    return value
                else:
                    print(f"❌ Key {key} not found in {table_name}")
                    return None
    
    def delete(self, table_name: str, key: str, transaction_id: Optional[int] = None) -> bool:
        """Delete a record"""
        if transaction_id is None:
            # Direct operation
            with self.lock:
                if table_name in self.data and key in self.data[table_name]:
                    del self.data[table_name][key]
                    print(f"✅ Deleted {key} from {table_name}")
                    return True
                else:
                    print(f"❌ Key {key} not found in {table_name}")
                    return False
        else:
            # Transaction operation
            if transaction_id not in self.transactions:
                print(f"❌ Transaction {transaction_id} not found")
                return False
            
            transaction = self.transactions[transaction_id]
            if transaction.status != TransactionStatus.ACTIVE:
                print(f"❌ Transaction {transaction_id} is not active")
                return False
            
            if table_name not in transaction.changes:
                transaction.changes[table_name] = {}
            transaction.changes[table_name][key] = None  # Mark for deletion
            print(f"📝 Transaction {transaction_id}: Staged delete {key} from {table_name}")
            return True
    
    def begin_transaction(self) -> int:
        """Start a new transaction"""
        with self.lock:
            transaction_id = self.next_transaction_id
            self.next_transaction_id += 1
            
            transaction = Transaction(
                id=transaction_id,
                status=TransactionStatus.ACTIVE,
                changes={},
                start_time=time.time()
            )
            self.transactions[transaction_id] = transaction
            print(f"🔄 Started transaction {transaction_id}")
            return transaction_id
    
    def commit_transaction(self, transaction_id: int) -> bool:
        """Commit a transaction"""
        if transaction_id not in self.transactions:
            print(f"❌ Transaction {transaction_id} not found")
            return False
        
        transaction = self.transactions[transaction_id]
        if transaction.status != TransactionStatus.ACTIVE:
            print(f"❌ Transaction {transaction_id} is not active")
            return False
        
        # Apply all changes atomically
        with self.lock:
            for table_name, changes in transaction.changes.items():
                if table_name not in self.data:
                    self.data[table_name] = {}
                
                for key, value in changes.items():
                    if value is None:
                        # Delete operation
                        if key in self.data[table_name]:
                            del self.data[table_name][key]
                    else:
                        # Insert/Update operation
                        self.data[table_name][key] = value
            
            transaction.status = TransactionStatus.COMMITTED
            print(f"✅ Transaction {transaction_id} committed successfully")
            return True
    
    def rollback_transaction(self, transaction_id: int) -> bool:
        """Rollback a transaction"""
        if transaction_id not in self.transactions:
            print(f"❌ Transaction {transaction_id} not found")
            return False
        
        transaction = self.transactions[transaction_id]
        if transaction.status != TransactionStatus.ACTIVE:
            print(f"❌ Transaction {transaction_id} is not active")
            return False
        
        transaction.status = TransactionStatus.ABORTED
        print(f"🔄 Transaction {transaction_id} rolled back")
        return True
    
    def show_data(self, table_name: Optional[str] = None) -> None:
        """Show current data state"""
        print("\n📊 Current Database State:")
        print("=" * 50)
        
        if table_name:
            if table_name in self.data:
                print(f"Table '{table_name}':")
                for key, value in self.data[table_name].items():
                    print(f"  {key}: {value}")
            else:
                print(f"Table '{table_name}' does not exist")
        else:
            for table_name, table_data in self.data.items():
                print(f"Table '{table_name}':")
                for key, value in table_data.items():
                    print(f"  {key}: {value}")
        
        print("=" * 50)
    
    def show_transactions(self) -> None:
        """Show active transactions"""
        print("\n🔄 Active Transactions:")
        print("=" * 50)
        
        active_transactions = [t for t in self.transactions.values() if t.status == TransactionStatus.ACTIVE]
        
        if not active_transactions:
            print("No active transactions")
        else:
            for transaction in active_transactions:
                print(f"Transaction {transaction.id}:")
                for table_name, changes in transaction.changes.items():
                    print(f"  Table '{table_name}':")
                    for key, value in changes.items():
                        if value is None:
                            print(f"    DELETE {key}")
                        else:
                            print(f"    {key}: {value}")
        
        print("=" * 50)

