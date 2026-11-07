pub mod cli;
pub mod detok;
pub mod grammar;
pub mod reply;
pub mod template;
pub mod tokenizer;

use crate::error::Error;
use crate::json::Json;
use crate::model::grammar::THINK;
use crate::model::reply::{Calls, Opening, Reader};
use crate::model::template::{Choice, Conversation};

pub const NAME: &str = "deepseek-v4.1-flash";

pub struct Chat {
    pub prompt: String,
    pub reader: Reader,
}

pub fn chat(body: &Json) -> Result<Chat, Error> {
    let conversation = Conversation::parse(body)?;
    let prompt = conversation.render();
    Ok(Chat { reader: reader(&conversation, &prompt), prompt })
}

fn reader(conversation: &Conversation, prompt: &str) -> Reader {
    let opening = if prompt.ends_with(THINK) { Opening::Reasoning } else { Opening::Content };
    let calls = if conversation.choice != Choice::None && conversation.tools().next().is_some() { Calls::Read } else { Calls::Ignored };
    let mut reader = Reader::new(opening, calls);
    let primed = reader.push(&conversation.choice.forced());
    debug_assert!(primed.is_empty(), "{primed:?}");
    reader
}
