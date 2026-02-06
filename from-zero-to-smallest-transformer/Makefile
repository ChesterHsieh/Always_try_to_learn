.PHONY: help build run test check clippy fmt clean release

help:
	@echo "可用的命令："
	@echo "  make build      - 编译项目 (debug 模式)"
	@echo "  make run        - 运行项目"
	@echo "  make test       - 运行测试"
	@echo "  make check      - 快速检查代码"
	@echo "  make clippy     - 运行 clippy 代码检查"
	@echo "  make fmt        - 格式化代码"
	@echo "  make clean      - 清理构建文件"
	@echo "  make release    - 编译优化版本并运行"

build:
	cargo build

run:
	cargo run

test:
	cargo test

check:
	cargo check

clippy:
	cargo clippy

fmt:
	cargo fmt

clean:
	cargo clean

release:
	cargo build --release
	@echo "\n运行优化版本："
	./target/release/transformer
