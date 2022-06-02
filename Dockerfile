FROM konstantinschulz/bert-base-german-cased:latest as GBERT
FROM python:3.9-slim
RUN apt-get update
# avoid https://github.com/debuerreotype/debuerreotype/issues/10
RUN mkdir -p /usr/share/man/man1
# Install tini and create an unprivileged user
ADD https://github.com/krallin/tini/releases/download/v0.19.0/tini /sbin/tini
RUN addgroup --gid 1001 "elg" && adduser --disabled-password --gecos "ELG User,,," --home /elg --ingroup elg --uid 1001 elg && chmod +x /sbin/tini
# Everything from here down runs as the unprivileged user account
USER elg:elg
WORKDIR /elg
ENV WORKERS=1
# Create a Python virtual environment for the dependencies
RUN python -m venv venv
RUN /elg/venv/bin/pip --no-cache-dir install torch==1.11.0
COPY --from=GBERT /workspace /elg/models
# Copy in our app, its requirements file and the entrypoint script
COPY --chown=elg:elg models/ requirements.txt docker-entrypoint.sh /elg/
COPY src/elg_service.py /elg/src
RUN /elg/venv/bin/pip --no-cache-dir install -r requirements.txt
COPY --chown=elg:elg . .
RUN chmod +x ./docker-entrypoint.sh
ENTRYPOINT ["./docker-entrypoint.sh"]