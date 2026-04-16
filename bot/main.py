import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters

from llm.engine import process_message, format_result_singlish
from cache.session import SessionCache
from guards.topic_check import quick_topic_check
from services.backend_client import call_predict

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

session_cache = SessionCache(
    maxsize=int(os.environ.get("SESSION_MAXSIZE", 500)),
    ttl=int(os.environ.get("SESSION_TTL", 3600)),
)

ptb_app: Application = None


# ── Telegram handlers ────────────────────────────────────────────────────────

async def handle_message(update: Update, context) -> None:
    chat_id = update.effective_chat.id
    user_msg = update.message.text.strip()

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    quick_topic_check(user_msg)

    state = session_cache.get(chat_id)
    llm_result = await process_message(
        user_message=user_msg,
        conversation_history=state["history"],
        collected_params=state["collected_params"],
    )

    session_cache.append_history(chat_id, "user", user_msg)
    session_cache.append_history(chat_id, "assistant", llm_result["reply"])
    if llm_result.get("extracted_params"):
        session_cache.merge_params(chat_id, llm_result["extracted_params"])

    await update.message.reply_text(llm_result["reply"], parse_mode="Markdown")

    if llm_result.get("ready_to_predict") and session_cache.is_complete(chat_id):
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        updated_state = session_cache.get(chat_id)
        try:
            prediction = await call_predict(updated_state["collected_params"])
            singlish_reply = await format_result_singlish(
                prediction, updated_state["collected_params"]
            )
            await update.message.reply_text(singlish_reply, parse_mode="Markdown")
        except Exception as exc:
            logger.error(f"Prediction call failed: {exc}")
            await update.message.reply_text(
                "Aiyoh, something went wrong when I try to calculate leh 😅\n"
                "Try again? Type /estimate to start fresh."
            )
        finally:
            session_cache.clear(chat_id)


async def cmd_start(update: Update, context) -> None:
    session_cache.clear(update.effective_chat.id)
    await update.message.reply_text(
        "Eh hello! 👋 I'm Uncle HDB — your kakak for checking HDB resale prices in SG!\n\n"
        "Just tell me about the flat lor — which area, what type, high or low floor, liddat. "
        "I'll figure out the rest.\n\n"
        "So, what flat you want to check ah? 🏠",
        parse_mode="Markdown",
    )


async def cmd_cancel(update: Update, context) -> None:
    session_cache.clear(update.effective_chat.id)
    await update.message.reply_text(
        "Ok lor, I clear everything already 👌\n"
        "Whenever ready, just /estimate and we start fresh can!"
    )


async def cmd_help(update: Update, context) -> None:
    await update.message.reply_text(
        "🏠 *Uncle HDB Help*\n\n"
        "Just chat with me about the HDB flat you want to check!\n"
        "Tell me the town, flat type, model, floor, area, lease, street and block.\n"
        "Can give all at once or one by one, up to you lor.\n\n"
        "Commands:\n"
        "/estimate — Start a new price check\n"
        "/cancel — Clear and start over\n"
        "/help — Show this message",
        parse_mode="Markdown",
    )


# ── Bot initialisation (runs in background after server starts) ──────────────

async def _init_bot():
    global ptb_app
    try:
        app = Application.builder().token(os.environ["TELEGRAM_BOT_TOKEN"]).build()
        app.add_handler(CommandHandler("start",    cmd_start))
        app.add_handler(CommandHandler("estimate", cmd_start))
        app.add_handler(CommandHandler("cancel",   cmd_cancel))
        app.add_handler(CommandHandler("help",     cmd_help))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        await app.initialize()
        ptb_app = app
        logger.info("Telegram bot initialised ✅")
    except Exception as exc:
        logger.error(f"Bot initialisation failed: {exc}")


# ── FastAPI app ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start bot init in background so port 8080 binds immediately
    asyncio.create_task(_init_bot())
    yield
    if ptb_app:
        await ptb_app.shutdown()


web_app = FastAPI(lifespan=lifespan)


@web_app.get("/health")
async def health():
    return {"status": "ok", "bot_ready": ptb_app is not None}


@web_app.post("/webhook")
async def webhook(request: Request):
    if ptb_app is None:
        return Response(status_code=503)
    data = await request.json()
    update = Update.de_json(data, ptb_app.bot)
    await ptb_app.process_update(update)
    return Response(status_code=200)
