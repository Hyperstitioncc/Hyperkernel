use crate::json::Json;

pub const BOS: &str = "<｜begin▁of▁sentence｜>";
pub const EOS: &str = "<｜end▁of▁sentence｜>";
pub const SYSTEM: &str = "<｜System｜>";
pub const USER: &str = "<｜User｜>";
pub const ASSISTANT: &str = "<｜Assistant｜>";
pub const LATEST_REMINDER: &str = "<｜latest_reminder｜>";
pub const THINK: &str = "<think>";
pub const END_THINK: &str = "</think>";

macro_rules! calls {
    () => {
        "｜DSML｜ calls"
    };
}
macro_rules! invoke {
    () => {
        "｜DSML｜ invoke"
    };
}
macro_rules! parameter {
    () => {
        "｜DSML｜ parameter"
    };
}

pub const CALLS: &str = calls!();
pub const INVOKE: &str = invoke!();
pub const PARAMETER: &str = parameter!();

pub const BLOCK_START: &str = concat!("\n\n<", calls!());
const BLOCK_OPEN: &str = concat!("\n\n<", calls!(), ">");
const BLOCK_END: &str = concat!("\n</", calls!(), ">");
const INVOKE_OPEN: &str = concat!("\n<", invoke!());
const INVOKE_END: &str = concat!("</", invoke!(), ">");
const PARAMETER_OPEN: &str = concat!("<", parameter!());
const VALUE_END: &str = concat!("/", parameter!());
const NEXT_PARAMETER: &str = concat!(">\n<", parameter!());
const NEXT_INVOKE_END: &str = concat!(">\n</", invoke!(), ">");
const STRING: &str = "\" string=\"true\">";
const NOT_STRING: &str = "\" string=\"false\">";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Param {
    pub key: String,
    pub value: String,
    pub string: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Invoke {
    pub name: String,
    pub params: Vec<Param>,
}

pub enum Item {
    Call(Invoke),
    End,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Miss {
    Wait,
    Broken,
}

impl Param {
    pub fn of(key: &str, value: &Json) -> Param {
        match value.as_str() {
            Some(s) => Param { key: key.to_owned(), value: s.to_owned(), string: true },
            None => Param { key: key.to_owned(), value: value.python(), string: false },
        }
    }

    fn write(&self) -> String {
        format!("<{PARAMETER} name=\"{}\" string=\"{}\">{}</{PARAMETER}>", self.key, self.string, self.value)
    }

    fn read(body: &str) -> Option<Param> {
        let rest = body.strip_prefix(" name=\"")?.strip_suffix('<')?;
        let (key, marker, value) = split_first(rest, &[STRING, NOT_STRING])?;
        Some(Param { key: key.to_owned(), value: value.to_owned(), string: marker == STRING })
    }

    fn json(&self) -> String {
        if self.string { Json::String(self.value.clone()).python() } else { self.value.clone() }
    }
}

impl Invoke {
    fn write(&self) -> String {
        let params: Vec<String> = self.params.iter().map(Param::write).collect();
        format!("<{INVOKE} name=\"{}\">\n{}\n</{INVOKE}>", self.name, params.join("\n"))
    }

    pub fn arguments(&self) -> String {
        let members: Vec<String> = self.params.iter().map(|p| format!("{}: {}", Json::String(p.key.clone()).python(), p.json())).collect();
        format!("{{{}}}", members.join(", "))
    }
}

pub fn write_block<'c>(calls: impl IntoIterator<Item = &'c Invoke>) -> String {
    let calls: Vec<String> = calls.into_iter().map(Invoke::write).collect();
    format!("{BLOCK_OPEN}\n{}{BLOCK_END}", calls.join("\n"))
}

pub fn forced_block(name: Option<&str>) -> String {
    match name {
        Some(name) => format!("{BLOCK_OPEN}{INVOKE_OPEN} name=\"{name}\">\n"),
        None => format!("{BLOCK_OPEN}{INVOKE_OPEN} name=\""),
    }
}

pub fn read_first(text: &str) -> Result<(Item, usize), Miss> {
    literal(text, &[BLOCK_OPEN])?;
    let (item, n) = read_item(&text[BLOCK_OPEN.len()..])?;
    Ok((item, BLOCK_OPEN.len() + n))
}

pub fn read_item(text: &str) -> Result<(Item, usize), Miss> {
    let mut rest = text;
    if literal_in(&mut rest, &[INVOKE_OPEN, BLOCK_END])? == BLOCK_END {
        return Ok((Item::End, text.len() - rest.len()));
    }
    let (head, mut next) = until_in(&mut rest, &[PARAMETER_OPEN, INVOKE_END])?;
    let name = head.trim_start().strip_prefix("name=\"").and_then(|h| h.strip_suffix("\">\n")).ok_or(Miss::Broken)?;
    let mut params: Vec<Param> = Vec::new();
    while next == PARAMETER_OPEN {
        let (body, _) = until_in(&mut rest, &[VALUE_END])?;
        let param = Param::read(body).filter(|p| params.iter().all(|q| q.key != p.key)).ok_or(Miss::Broken)?;
        params.push(param);
        next = if literal_in(&mut rest, &[NEXT_PARAMETER, NEXT_INVOKE_END])? == NEXT_PARAMETER { PARAMETER_OPEN } else { INVOKE_END };
    }
    Ok((Item::Call(Invoke { name: name.to_owned(), params }), text.len() - rest.len()))
}

fn literal(text: &str, literals: &[&'static str]) -> Result<&'static str, Miss> {
    match literals.iter().find(|l| text.starts_with(*l)) {
        Some(&l) => Ok(l),
        None if literals.iter().any(|l| l.starts_with(text)) => Err(Miss::Wait),
        None => Err(Miss::Broken),
    }
}

fn literal_in(rest: &mut &str, literals: &[&'static str]) -> Result<&'static str, Miss> {
    let l = literal(rest, literals)?;
    *rest = &rest[l.len()..];
    Ok(l)
}

fn until_in<'t>(rest: &mut &'t str, markers: &[&'static str]) -> Result<(&'t str, &'static str), Miss> {
    let (before, marker, after) = split_first(rest, markers).ok_or(Miss::Wait)?;
    *rest = after;
    Ok((before, marker))
}

fn split_first<'t>(text: &'t str, markers: &[&'static str]) -> Option<(&'t str, &'static str, &'t str)> {
    let (at, marker) = markers.iter().filter_map(|&m| text.find(m).map(|at| (at, m))).min()?;
    Some((&text[..at], marker, &text[at + marker.len()..]))
}
