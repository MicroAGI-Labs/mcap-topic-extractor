"""
List all topics in an MCAP file.

Usage:
    python list_mcap_topics.py <mcap_file>
"""

import sys
from mcap.reader import make_reader


def list_topics(mcap_path: str):
    print(f"\nScanning: {mcap_path}\n")
    
    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        
        if summary and summary.statistics:
            stats = summary.statistics
            duration = (stats.message_end_time - stats.message_start_time) / 1e9
            print(f"Duration: {duration:.2f}s | Messages: {stats.message_count}")
        
        print("-" * 60)
        
        # Get channels and schemas
        channels = summary.channels if summary else {}
        schemas = summary.schemas if summary else {}
        
        for channel_id, channel in sorted(channels.items(), key=lambda x: x[1].topic):
            schema = schemas.get(channel.schema_id)
            schema_name = schema.name if schema else "unknown"
            msg_count = summary.statistics.channel_message_counts.get(channel_id, 0) if summary and summary.statistics else "?"
            print(f"{channel.topic}: {msg_count} msgs ({schema_name})")
        
        print("-" * 60)
        print(f"Total: {len(channels)} topics")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python list_mcap_topics.py <mcap_file>")
        sys.exit(1)
    list_topics(sys.argv[1])

