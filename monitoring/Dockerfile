# ====================================================================== #
# Dockerfile untuk menjalankan Prometheus pada tahap monitoring.
# ====================================================================== #
FROM prom/prometheus:v2.48.1

# Salin konfigurasi Prometheus
COPY prometheus.yml /etc/prometheus/prometheus.yml
COPY prometheus.config /etc/prometheus/prometheus.config

# Persistensi data Prometheus
VOLUME ["/prometheus"]

# Port Prometheus web UI
EXPOSE 9090

# Command default dari image resmi sudah memuat config.yml,
# di sini kita override agar memakai prometheus.yml yang sudah disalin.
CMD ["--config.file=/etc/prometheus/prometheus.yml", "--storage.tsdb.path=/prometheus", "--web.console.libraries=/usr/share/prometheus/console_libraries", "--web.console.templates=/usr/share/prometheus/consoles", "--web.enable-lifecycle"]
ENTRYPOINT ["/bin/prometheus"]
