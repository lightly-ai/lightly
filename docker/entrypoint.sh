#!/bin/bash

for DIR in /home/pretrain/output_dir; do
    chown -R pretrain "$DIR" 2>/dev/null || true
    chmod -R g+s "$DIR" 2>/dev/null || true
    setfacl -R -m g::rwX "$DIR" 2>/dev/null || true
    setfacl -d -m g::rwX "$DIR" 2>/dev/null || true
done

tensorboard --logdir /home/pretrain/output_dir --host 0.0.0.0 --port 6006 --load_fast=false &
exec lightly-pretrain "$@"
